
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class HyperLoRAGenerator(nn.Module):    
    def __init__(
        self,
        proto_dim: int,
        lora_rank: int,
        in_features: int,
        out_features: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.proto_dim = proto_dim
        self.lora_rank = lora_rank
        self.in_features = in_features
        self.out_features = out_features
        
        self.gen_lora_a = nn.Sequential(
            nn.Linear(proto_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, lora_rank * in_features),
        )
        
        self.gen_lora_b = nn.Sequential(
            nn.Linear(proto_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features * lora_rank),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, proto_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_output = False
        if proto_feature.dim() == 1:
            proto_feature = proto_feature.unsqueeze(0)
            squeeze_output = True
        
        batch_size = proto_feature.shape[0]
        
        lora_a_flat = self.gen_lora_a(proto_feature)
        lora_a = lora_a_flat.view(batch_size, self.lora_rank, self.in_features)
        
        lora_b_flat = self.gen_lora_b(proto_feature)
        lora_b = lora_b_flat.view(batch_size, self.out_features, self.lora_rank)
        
        if squeeze_output:
            lora_a = lora_a.squeeze(0)
            lora_b = lora_b.squeeze(0)
        
        return lora_a, lora_b


class GlobalHyperNetwork(nn.Module):
    
    def __init__(
        self,
        proto_dim: int,
        lora_rank: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.proto_dim = proto_dim
        self.lora_rank = lora_rank
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.generators = nn.ModuleDict()
        self._current_weighted_proto = None
        
        self._lora_cache = {}
    
    def register_layer(self, layer_name: str, in_features: int, out_features: int, dropout: float = 0.1):
        key = f"{in_features}_{out_features}"
        if key not in self.generators:
            self.generators[key] = HyperLoRAGenerator(
                proto_dim=self.proto_dim,
                lora_rank=self.lora_rank,
                in_features=in_features,
                out_features=out_features,
                hidden_dim=self.hidden_dim,
                dropout=dropout
            )
            print(f"[GlobalHyperNetwork] Registered generator for size {key}")
    
    def set_weighted_proto(self, weighted_proto):
        self._lora_cache = {}
        
        if weighted_proto is None:
            self._current_weighted_proto = None
            return
        
        self._current_weighted_proto = weighted_proto.detach()
        
        for key, generator in self.generators.items():
            lora_a, lora_b = generator(self._current_weighted_proto)
            self._lora_cache[key] = (lora_a.detach(), lora_b.detach())
    
    def generate_for_layer(self, layer_key: str):
        if layer_key in self._lora_cache:
            return self._lora_cache[layer_key]
        return None


class DynamicLoRALinear(nn.Module):
    
    def __init__(
        self,
        base_layer: nn.Linear,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        self.weight = nn.Parameter(base_layer.weight.detach().clone(), requires_grad=False)
        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        self._layer_key = f"{self.in_features}_{self.out_features}"
        self._global_hypernet = None
    
    def set_global_hypernet(self, hypernet):
        self._global_hypernet = hypernet
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = F.linear(x, self.weight, self.bias)
        
        if self._global_hypernet is None:
            return base_output
        
        lora_params = self._global_hypernet.generate_for_layer(self._layer_key)
        if lora_params is None:
            return base_output
        
        lora_a, lora_b = lora_params
        
        original_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        
        h = F.linear(x_2d, lora_a)
        h = self.lora_dropout(h)
        lora_output = F.linear(h, lora_b)
        
        lora_output = lora_output.view(*original_shape[:-1], -1)
        
        return base_output + lora_output * self.scaling


def replace_linear_with_hyperlora(
    model: nn.Module,
    proto_lib,
    target_modules: list,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0
):
    
    global_hypernet = GlobalHyperNetwork(
        proto_dim=proto_lib.proto_dim,
        lora_rank=lora_rank,
        hidden_dim=256,
        dropout=lora_dropout
    )
    
    layers_to_replace = []
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                layers_to_replace.append((name, module))
    
    print(f"[ProtoStream] Found {len(layers_to_replace)} layers to replace")
    
    dynamic_layers = []
    
    for name, module in layers_to_replace:
        global_hypernet.register_layer(name, module.in_features, module.out_features, lora_dropout)
        
        dynamic_layer = DynamicLoRALinear(
            base_layer=module,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        dynamic_layers.append(dynamic_layer)
        
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        parent = model
        if parent_name:
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
        
        setattr(parent, child_name, dynamic_layer)
    
    print(f"[ProtoStream] Replaced {len(dynamic_layers)} layers with DynamicLoRALinear")
    
    model.add_module('global_hypernet', global_hypernet)
    
    for layer in dynamic_layers:
        layer.set_global_hypernet(global_hypernet)
    
    return len(dynamic_layers)