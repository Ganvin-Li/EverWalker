
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SceneEncoder(nn.Module):
    
    def __init__(
        self,
        vision_tower,
        text_encoder=None,
        output_dim: int = 512,
        use_text: bool = False
    ):

        super().__init__()
        
        self.__dict__['vision_tower'] = vision_tower
        self.__dict__['text_encoder'] = text_encoder
        self.output_dim = output_dim
        self.use_text = use_text
        
        if hasattr(vision_tower, 'hidden_size'):
            vision_dim = vision_tower.hidden_size
        elif hasattr(vision_tower, 'config') and hasattr(vision_tower.config, 'hidden_size'):
            vision_dim = vision_tower.config.hidden_size
        else:
            vision_dim = 1024
        
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        if use_text and text_encoder is not None:
            if hasattr(text_encoder, 'hidden_size'):
                text_dim = text_encoder.hidden_size
            elif hasattr(text_encoder, 'config') and hasattr(text_encoder.config, 'hidden_size'):
                text_dim = text_encoder.config.hidden_size
            else:
                text_dim = 4096  
            
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.LayerNorm(output_dim)
            )
        
        for param in self.vision_tower.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        images: torch.Tensor,
        instruction_text: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size = images.shape[0]
        
        if images.dim() == 5:
            num_frames = images.shape[1]
            images = images.reshape(-1, *images.shape[2:])
        else:
            num_frames = 1
        
        with torch.no_grad():
            if hasattr(self.vision_tower, 'forward'):
                vision_feats = self.vision_tower(images)
            else:
                vision_feats = self.vision_tower.encode_image(images)
        
        if isinstance(vision_feats, tuple):
            vision_feats = vision_feats[0]
        
        if vision_feats.dim() == 2:
            vision_feats = vision_feats.unsqueeze(1)
        
        vision_feats = vision_feats.mean(dim=1)
        
        if num_frames > 1:
            vision_feats = vision_feats.reshape(batch_size, num_frames, -1)
            vision_feats = vision_feats.mean(dim=1)
        
        scene_repr = self.vision_proj(vision_feats)  # (B, output_dim)
        
        if self.use_text and instruction_text is not None and self.text_encoder is not None:
            text_feats = self.text_encoder(instruction_text)
            
            if isinstance(text_feats, tuple):
                text_feats = text_feats[0]
            
            if text_feats.dim() == 3:
                text_feats = text_feats.mean(dim=1)
            
            text_feats = self.text_proj(text_feats)  # (B, output_dim)
            
            combined = torch.cat([scene_repr, text_feats], dim=-1)
            scene_repr = self.fusion(combined)
        
        scene_repr = F.normalize(scene_repr, p=2, dim=-1)
        
        return scene_repr
    
    def extract_batch_features(self, image_batch: torch.Tensor) -> torch.Tensor:

        return self.forward(image_batch)


class SceneFeatureCache:

    def __init__(self, max_size: int = 1000):

        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key: str) -> Optional[torch.Tensor]:

        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, key: str, feature: torch.Tensor):

        if len(self.cache) >= self.max_size:
            min_key = min(self.access_count, key=self.access_count.get)
            del self.cache[min_key]
            del self.access_count[min_key]
        
        self.cache[key] = feature.detach().clone()
        self.access_count[key] = 1
    
    def clear(self):
        self.cache.clear()
        self.access_count.clear()
    
    def __len__(self):
        return len(self.cache)
