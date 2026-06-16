
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import numpy as np
import os


class PrototypeLibrary(nn.Module):

    
    def __init__(
        self,
        proto_dim: int = 512,
        similarity_threshold: float = 0.75, 
        momentum: float = 0.95,  
        max_prototypes: int = 100,
        diversity_weight: float = 0.1,
        temperature: float = 1.0
    ):

        super().__init__()
        
        self.proto_dim = proto_dim
        self.threshold = similarity_threshold
        self.momentum = momentum
        self.max_prototypes = max_prototypes
        self.diversity_weight = diversity_weight
        self.temperature = temperature
        
        self.prototypes = nn.ParameterList()
        self.proto_counts = []  
        
        self.frozen_prototypes = []  # List[List[Tensor]]
        self.frozen_proto_counts = []  # List[List[int]]
        
        self.total_samples = 0
        self.creation_history = [] 
    
    @property
    def device(self) -> torch.device:

        if len(self.prototypes) > 0:
            return self.prototypes[0].device
        else:
            return torch.device('cpu')
    
    def to(self, device):
        super().to(device)
        for task_protos in self.frozen_prototypes:
            for i, proto in enumerate(task_protos):
                task_protos[i] = proto.to(device)
        return self
    
    def match_prototype(
        self,
        scene_features: torch.Tensor,
        return_all_similarities: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if len(self.prototypes) == 0:
            return None, None
        
        proto_matrix = torch.stack([p for p in self.prototypes], dim=0)
        
        scene_features = F.normalize(scene_features, p=2, dim=-1)
        proto_matrix = F.normalize(proto_matrix, p=2, dim=-1)
        
        sim_matrix = torch.mm(scene_features, proto_matrix.T)
        
        if return_all_similarities:
            max_sims, proto_indices = sim_matrix.max(dim=-1)
            return proto_indices, sim_matrix
        else:
            max_sims, proto_indices = sim_matrix.max(dim=-1)
            return proto_indices, max_sims
    
    def compute_soft_weights(
        self,
        scene_features: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if len(self.prototypes) == 0:
            return None, None
        
        _, sim_matrix = self.match_prototype(
            scene_features,
            return_all_similarities=True
        )
        
        sim_matrix = sim_matrix / self.temperature
        weights = F.softmax(sim_matrix, dim=-1)
        
        if top_k is not None and top_k < len(self.prototypes):
            top_weights, top_indices = torch.topk(weights, k=top_k, dim=-1)
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
            return top_indices, top_weights
        else:
            indices = torch.arange(len(self.prototypes), device=weights.device)
            indices = indices.unsqueeze(0).expand(weights.shape[0], -1)
            return indices, weights
    
    def update_or_create_prototype(
        self,
        scene_features: torch.Tensor,
        proto_idx: Optional[torch.Tensor] = None,
        similarity: Optional[torch.Tensor] = None,
        force_create: bool = False
    ) -> torch.Tensor:

        if scene_features.dim() == 1:
            scene_features = scene_features.unsqueeze(0)
        
        batch_size = scene_features.shape[0]
        scene_features = F.normalize(scene_features, p=2, dim=-1)
        
        if force_create or len(self.prototypes) == 0:
            result_indices = []
            for i in range(batch_size):
                new_idx = self._create_new_prototype(scene_features[i])
                result_indices.append(new_idx)
            return torch.tensor(result_indices, device=scene_features.device)
        
        result_indices = []
        for i in range(batch_size):
            feat = scene_features[i]
            
            if proto_idx is not None and similarity is not None:
                idx = proto_idx[i].item() if proto_idx.dim() > 0 else proto_idx.item()
                sim = similarity[i].item() if similarity.dim() > 0 else similarity.item()
                
                if sim < self.threshold:
                    new_idx = self._create_new_prototype(feat)
                    result_indices.append(new_idx)
                else:
                    self._update_prototype(idx, feat)
                    result_indices.append(idx)
            else:
                idx, sim = self.match_prototype(feat.unsqueeze(0))
                if idx is None or sim is None or sim.item() < self.threshold:
                    new_idx = self._create_new_prototype(feat)
                    result_indices.append(new_idx)
                else:
                    idx = idx.item()
                    self._update_prototype(idx, feat)
                    result_indices.append(idx)
        
        self.total_samples += batch_size
        return torch.tensor(result_indices, device=scene_features.device)
    
    def _create_new_prototype(self, feature: torch.Tensor) -> int:
        if len(self.prototypes) >= self.max_prototypes:
            self._merge_closest_prototypes()
        
        new_proto = nn.Parameter(feature.detach().clone())
        self.prototypes.append(new_proto)
        self.proto_counts.append(1)
        self.creation_history.append(self.total_samples)
        
        new_idx = len(self.prototypes) - 1
        print(f"[ProtoLib] Created prototype #{new_idx}, total: {len(self.prototypes)}")
        
        return new_idx
    
    def _update_prototype(self, proto_idx: int, feature: torch.Tensor):
        with torch.no_grad():
            old_proto = self.prototypes[proto_idx].data
            new_proto = self.momentum * old_proto + (1 - self.momentum) * feature
            self.prototypes[proto_idx].data = F.normalize(new_proto, p=2, dim=-1)
        
        self.proto_counts[proto_idx] += 1
    
    def _merge_closest_prototypes(self):

        if len(self.prototypes) < 2:
            return
        
        proto_matrix = torch.stack([p for p in self.prototypes], dim=0)
        proto_matrix = F.normalize(proto_matrix, p=2, dim=-1)
        sim_matrix = torch.mm(proto_matrix, proto_matrix.T)
        
        sim_matrix.fill_diagonal_(-1)
        
        max_sim = sim_matrix.max()
        idx1, idx2 = (sim_matrix == max_sim).nonzero(as_tuple=False)[0].tolist()
        
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        count1 = self.proto_counts[idx1]
        count2 = self.proto_counts[idx2]
        total_count = count1 + count2
        
        merged_proto = (
            count1 / total_count * self.prototypes[idx1].data +
            count2 / total_count * self.prototypes[idx2].data
        )
        merged_proto = F.normalize(merged_proto, p=2, dim=-1)
        
        new_prototypes = []
        new_counts = []
        new_history = []
        
        for i in range(len(self.prototypes)):
            if i == idx1:
                new_prototypes.append(nn.Parameter(merged_proto.clone()))
                new_counts.append(total_count)
                new_history.append(self.creation_history[i])
            elif i == idx2:
                continue
            else:
                new_prototypes.append(nn.Parameter(self.prototypes[i].data.clone()))
                new_counts.append(self.proto_counts[i])
                new_history.append(self.creation_history[i])
        
        self.prototypes = nn.ParameterList(new_prototypes)
        self.proto_counts = new_counts
        self.creation_history = new_history
        
        print(f"[ProtoLib] Merged prototypes {idx1} and {idx2}, remaining: {len(self.prototypes)}")
    
    def compute_diversity_loss(self) -> torch.Tensor:
        if len(self.prototypes) < 2:
            device = self.device
            return torch.tensor(0.0, device=device)
        
        proto_matrix = torch.stack([p for p in self.prototypes], dim=0)
        proto_matrix = F.normalize(proto_matrix, p=2, dim=-1)
        
        sim_matrix = torch.mm(proto_matrix, proto_matrix.T)
        mask = 1 - torch.eye(len(self.prototypes), device=sim_matrix.device)
        avg_similarity = (sim_matrix * mask).sum() / (mask.sum() + 1e-8)
        
        diversity_loss = self.diversity_weight * avg_similarity.pow(2)
        
        return diversity_loss
    
    def freeze_current_prototypes(self):
        frozen_protos = [p.detach().clone() for p in self.prototypes]
        frozen_counts = self.proto_counts.copy()
        
        self.frozen_prototypes.append(frozen_protos)
        self.frozen_proto_counts.append(frozen_counts)
        
        print(f"[ProtoLib] Froze {len(frozen_protos)} prototypes for task {len(self.frozen_prototypes)}")
    
    def get_frozen_prototypes(self, task_id: Optional[int] = None) -> List[torch.Tensor]:
        if task_id is not None:
            if task_id < len(self.frozen_prototypes):
                return self.frozen_prototypes[task_id]
            else:
                return []
        else:
            all_protos = []
            for task_protos in self.frozen_prototypes:
                all_protos.extend(task_protos)
            return all_protos
    
    def get_statistics(self) -> Dict:
        if len(self.prototypes) == 0:
            return {
                'num_prototypes': 0,
                'total_samples': self.total_samples,
                'num_frozen_tasks': len(self.frozen_prototypes)
            }
        
        proto_matrix = torch.stack([p for p in self.prototypes], dim=0)
        proto_matrix = F.normalize(proto_matrix, p=2, dim=-1)
        sim_matrix = torch.mm(proto_matrix, proto_matrix.T)
        
        mask = 1 - torch.eye(len(self.prototypes), device=sim_matrix.device)
        avg_sim = (sim_matrix * mask).sum() / (mask.sum() + 1e-8)
        min_sim = (sim_matrix + torch.eye(len(self.prototypes), device=sim_matrix.device) * 10).min()
        
        return {
            'num_prototypes': len(self.prototypes),
            'total_samples': self.total_samples,
            'avg_proto_similarity': avg_sim.item(),
            'min_proto_similarity': min_sim.item(),
            'avg_proto_count': np.mean(self.proto_counts) if self.proto_counts else 0,
            'max_proto_count': max(self.proto_counts) if self.proto_counts else 0,
            'min_proto_count': min(self.proto_counts) if self.proto_counts else 0,
            'num_frozen_tasks': len(self.frozen_prototypes)
        }
    
    def get_all_prototypes(self) -> torch.Tensor:
        if len(self.prototypes) == 0:
            return torch.empty(0, self.proto_dim)
        return torch.stack([p for p in self.prototypes], dim=0)
        
    def state_dict_for_save(self) -> Dict:

        if len(self.prototypes) == 0:
            proto_tensor = torch.empty(0, self.proto_dim)
        else:
            proto_tensor = torch.stack([p.data for p in self.prototypes], dim=0)
        
        return {
            'prototypes': proto_tensor.cpu(),
            'proto_counts': self.proto_counts.copy(),
            'proto_dim': self.proto_dim,
            'threshold': self.threshold,
            'momentum': self.momentum,
            'max_prototypes': self.max_prototypes,
            'diversity_weight': self.diversity_weight,
            'temperature': self.temperature,
            'total_samples': self.total_samples,
            'creation_history': self.creation_history.copy(),
            'frozen_prototypes': [[p.cpu() for p in task_protos] for task_protos in self.frozen_prototypes],
            'frozen_proto_counts': [counts.copy() for counts in self.frozen_proto_counts],
            'num_prototypes': len(self.prototypes)
        }
    
    def load_state_dict_from_save(self, state_dict: Dict, device: torch.device = None):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.proto_dim = state_dict.get('proto_dim', self.proto_dim)
        self.threshold = state_dict.get('threshold', self.threshold)
        self.momentum = state_dict.get('momentum', self.momentum)
        self.max_prototypes = state_dict.get('max_prototypes', self.max_prototypes)
        self.diversity_weight = state_dict.get('diversity_weight', self.diversity_weight)
        self.temperature = state_dict.get('temperature', self.temperature)
        self.total_samples = state_dict.get('total_samples', 0)
        self.creation_history = state_dict.get('creation_history', [])
        
        proto_tensor = state_dict.get('prototypes', None)
        if proto_tensor is not None and proto_tensor.numel() > 0:
            num_protos = proto_tensor.shape[0]
            new_prototypes = []
            for i in range(num_protos):
                new_prototypes.append(nn.Parameter(proto_tensor[i].to(device)))
            self.prototypes = nn.ParameterList(new_prototypes)
        else:
            self.prototypes = nn.ParameterList()
        
        self.proto_counts = state_dict.get('proto_counts', [])
        if len(self.proto_counts) != len(self.prototypes):
            self.proto_counts = [1] * len(self.prototypes)
        
        frozen_protos = state_dict.get('frozen_prototypes', [])
        self.frozen_prototypes = []
        for task_protos in frozen_protos:
            self.frozen_prototypes.append([p.to(device) for p in task_protos])
        
        self.frozen_proto_counts = state_dict.get('frozen_proto_counts', [])
        
        print(f"[ProtoLib] Loaded {len(self.prototypes)} prototypes, "
              f"{len(self.frozen_prototypes)} frozen tasks")
    
    def save(self, path: str):
        state_dict = self.state_dict_for_save()
        torch.save(state_dict, path)
        print(f"[ProtoLib] Saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'PrototypeLibrary':
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        state_dict = torch.load(path, map_location='cpu')
        
        proto_lib = cls(
            proto_dim=state_dict.get('proto_dim', 512),
            similarity_threshold=state_dict.get('threshold', 0.75),
            momentum=state_dict.get('momentum', 0.95),
            max_prototypes=state_dict.get('max_prototypes', 100),
            diversity_weight=state_dict.get('diversity_weight', 0.1),
            temperature=state_dict.get('temperature', 1.0)
        )
        
        proto_lib.load_state_dict_from_save(state_dict, device)
        
        return proto_lib
    
    
    def sync_across_gpus(self):

        import torch.distributed as dist
        if not dist.is_initialized():
            return
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        local_num = torch.tensor([len(self.prototypes)], device='cuda')
        all_nums = [torch.zeros_like(local_num) for _ in range(world_size)]
        dist.all_gather(all_nums, local_num)
        
        master_num = all_nums[0].item()
        
        if master_num > 0:
            if rank == 0:
                proto_tensor = self.get_all_prototypes().cuda()
                counts_tensor = torch.tensor(self.proto_counts, dtype=torch.float32, device='cuda')
            else:
                proto_tensor = torch.zeros(master_num, self.proto_dim, device='cuda')
                counts_tensor = torch.zeros(master_num, device='cuda')
            
            dist.broadcast(proto_tensor, src=0)
            dist.broadcast(counts_tensor, src=0)
            
            if rank != 0:
                new_prototypes = []
                for i in range(master_num):
                    new_prototypes.append(nn.Parameter(proto_tensor[i].clone()))
                
                self.prototypes = nn.ParameterList(new_prototypes)
                self.proto_counts = counts_tensor.int().tolist()
                self.creation_history = list(range(master_num))
        
        print(f"[ProtoLib] Rank {rank}: Synced {len(self.prototypes)} prototypes")
    
    def __len__(self):
        return len(self.prototypes)
    
    def __repr__(self):
        return f"PrototypeLibrary(num_prototypes={len(self.prototypes)}, " \
               f"max={self.max_prototypes}, threshold={self.threshold})"

def save_protostream_checkpoint(
    output_dir: str,
    prototype_lib: PrototypeLibrary,
    scene_encoder: nn.Module = None,
    global_hypernet: nn.Module = None,
    cl_state: Dict = None,
    rank: int = 0
):

    if rank != 0:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    proto_lib_path = os.path.join(output_dir, "prototype_lib.pt")
    prototype_lib.save(proto_lib_path)
    
    if scene_encoder is not None:
        scene_encoder_path = os.path.join(output_dir, "scene_encoder.pt")
        torch.save(scene_encoder.state_dict(), scene_encoder_path)
        print(f"[ProtoStream] Scene encoder saved to {scene_encoder_path}")
    
    if global_hypernet is not None:
        hypernet_path = os.path.join(output_dir, "global_hypernet.pt")
        torch.save(global_hypernet.state_dict(), hypernet_path)
        print(f"[ProtoStream] Global hypernet saved to {hypernet_path}")
    
    if cl_state is not None:
        cl_state_path = os.path.join(output_dir, "cl_state.pt")
        torch.save(cl_state, cl_state_path)
        print(f"[ProtoStream] CL state saved to {cl_state_path}")
    
    stats = prototype_lib.get_statistics()
    stats_path = os.path.join(output_dir, "proto_stats.json")
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[ProtoStream] Stats saved to {stats_path}")
    
    print(f"[ProtoStream] Checkpoint saved to {output_dir}")


def load_protostream_checkpoint(
    checkpoint_dir: str,
    prototype_lib: PrototypeLibrary = None,
    scene_encoder: nn.Module = None,
    global_hypernet: nn.Module = None,
    device: torch.device = None
) -> Dict:

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    result = {
        'prototype_lib': None,
        'scene_encoder_loaded': False,
        'global_hypernet_loaded': False,
        'cl_state': None
    }
    
    print(f"[ProtoStream] Loading checkpoint from {checkpoint_dir}")
    
    proto_lib_path = os.path.join(checkpoint_dir, "prototype_lib.pt")
    if os.path.exists(proto_lib_path):
        try:
            if prototype_lib is None:
                prototype_lib = PrototypeLibrary.load(proto_lib_path, device)
            else:
                state_dict = torch.load(proto_lib_path, map_location='cpu')
                prototype_lib.load_state_dict_from_save(state_dict, device)
            result['prototype_lib'] = prototype_lib

    
    scene_encoder_path = os.path.join(checkpoint_dir, "scene_encoder.pt")
    if os.path.exists(scene_encoder_path) and scene_encoder is not None:
        try:
            state_dict = torch.load(scene_encoder_path, map_location=device)
            scene_encoder.load_state_dict(state_dict)
            result['scene_encoder_loaded'] = True
    
    hypernet_path = os.path.join(checkpoint_dir, "global_hypernet.pt")
    if os.path.exists(hypernet_path) and global_hypernet is not None:
            state_dict = torch.load(hypernet_path, map_location=device)
            global_hypernet.load_state_dict(state_dict)
            result['global_hypernet_loaded'] = True
    
    cl_state_path = os.path.join(checkpoint_dir, "cl_state.pt")
    if os.path.exists(cl_state_path):
        try:
            result['cl_state'] = torch.load(cl_state_path, map_location='cpu')
            print(f" Loaded cl_state")
        except Exception as e:
            print(f" Failed to load cl_state: {e}")
    
    print(f"[ProtoStream] Checkpoint loaded!")
    return result