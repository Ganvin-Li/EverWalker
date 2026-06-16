
import torch
import torch.nn as nn
import os
import json
from typing import Dict, Optional


def save_protostream_checkpoint(
    output_dir: str,
    model: nn.Module,
    rank: int = 0,
    save_hypernet: bool = True,
    save_scene_encoder: bool = True,
    save_prototype_lib: bool = True
):

    if rank != 0 and rank != -1:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"[ProtoStream] Saving checkpoint to {output_dir}")
    print(f"{'='*60}")
    
    if save_prototype_lib and hasattr(model, 'prototype_lib'):
        proto_lib_path = os.path.join(output_dir, "prototype_lib.pt")
        try:
            proto_lib = model.prototype_lib
            
            if hasattr(proto_lib, 'state_dict_for_save'):
                state_dict = proto_lib.state_dict_for_save()
            else:
                if len(proto_lib.prototypes) == 0:
                    proto_tensor = torch.empty(0, proto_lib.proto_dim)
                else:
                    proto_tensor = torch.stack([p.data for p in proto_lib.prototypes], dim=0)
                
                state_dict = {
                    'prototypes': proto_tensor.cpu(),
                    'proto_counts': proto_lib.proto_counts.copy() if hasattr(proto_lib, 'proto_counts') else [],
                    'proto_dim': proto_lib.proto_dim,
                    'threshold': proto_lib.threshold,
                    'momentum': proto_lib.momentum,
                    'max_prototypes': proto_lib.max_prototypes,
                    'total_samples': getattr(proto_lib, 'total_samples', 0),
                    'num_prototypes': len(proto_lib.prototypes)
                }
            
            torch.save(state_dict, proto_lib_path)
            print(f"  aved prototype_lib ({len(proto_lib)} prototypes)")
            
            if hasattr(proto_lib, 'get_statistics'):
                stats = proto_lib.get_statistics()
                stats_path = os.path.join(output_dir, "proto_stats.json")
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f" Saved proto_stats.json")
        except Exception as e:
            print(f" Failed to save prototype_lib: {e}")
    
    if save_scene_encoder and hasattr(model, 'scene_encoder'):
        scene_encoder_path = os.path.join(output_dir, "scene_encoder.pt")
        try:
            torch.save(model.scene_encoder.state_dict(), scene_encoder_path)
            print(f" Saved scene_encoder")
        except Exception as e:
            print(f" Failed to save scene_encoder: {e}")
    
    if save_hypernet and hasattr(model, 'global_hypernet'):
        hypernet_path = os.path.join(output_dir, "global_hypernet.pt")
        try:
            hypernet_state = {}
            for name, param in model.global_hypernet.state_dict().items():
                if 'base_layer' not in name:
                    hypernet_state[name] = param.cpu()
            
            torch.save(hypernet_state, hypernet_path)
            print(f" Saved global_hypernet ({len(hypernet_state)} params)")
        except Exception as e:
            print(f" Failed to save global_hypernet: {e}")
    
    cl_state_path = os.path.join(output_dir, "cl_state.pt")
    try:
        cl_state = {
            'has_prototype_lib': hasattr(model, 'prototype_lib'),
            'has_scene_encoder': hasattr(model, 'scene_encoder'),
            'has_global_hypernet': hasattr(model, 'global_hypernet'),
        }
        
        if hasattr(model, 'prototype_lib') and hasattr(model.prototype_lib, 'frozen_prototypes'):
            cl_state['num_frozen_tasks'] = len(model.prototype_lib.frozen_prototypes)
        
        torch.save(cl_state, cl_state_path)
        print(f" Saved cl_state.pt")
    except Exception as e:
        print(f" Failed to save cl_state: {e}")
    
    print(f"{'='*60}")
    print(f"[ProtoStream] Checkpoint saved successfully!")
    print(f"{'='*60}\n")


def load_protostream_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    device: torch.device = None,
    load_hypernet: bool = True,
    load_scene_encoder: bool = True,
    load_prototype_lib: bool = True
) -> Dict:

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    result = {
        'prototype_lib_loaded': False,
        'scene_encoder_loaded': False,
        'global_hypernet_loaded': False,
        'cl_state_loaded': False
    }
    
    print(f"\n{'='*60}")
    print(f"[Checkpoint] Loading from previous task: {checkpoint_dir}")
    print(f"{'='*60}")
    
    if load_scene_encoder and hasattr(model, 'scene_encoder'):
        scene_encoder_path = os.path.join(checkpoint_dir, "scene_encoder.pt")
        if os.path.exists(scene_encoder_path):
            try:
                state_dict = torch.load(scene_encoder_path, map_location=device)
                model.scene_encoder.load_state_dict(state_dict)
                result['scene_encoder_loaded'] = True
                print(f" Loaded scene_encoder ({len(state_dict)} params)")
            except Exception as e:
                print(f"  ailed to load scene_encoder: {e}")
    
    if load_hypernet and hasattr(model, 'global_hypernet'):
        hypernet_path = os.path.join(checkpoint_dir, "global_hypernet.pt")
        if os.path.exists(hypernet_path):
            try:
                state_dict = torch.load(hypernet_path, map_location=device)
                model.global_hypernet.load_state_dict(state_dict, strict=False)
                result['global_hypernet_loaded'] = True
            except Exception as e:
    
    if load_prototype_lib and hasattr(model, 'prototype_lib'):
        proto_lib_path = os.path.join(checkpoint_dir, "prototype_lib.pt")
        if os.path.exists(proto_lib_path):
            try:
                state_dict = torch.load(proto_lib_path, map_location='cpu')
                
                proto_lib = model.prototype_lib
                
                if hasattr(proto_lib, 'load_state_dict_from_save'):
                    proto_lib.load_state_dict_from_save(state_dict, device)
                else:
                    _load_prototype_lib_legacy(proto_lib, state_dict, device)
                
                result['prototype_lib_loaded'] = True
    
    cl_state_path = os.path.join(checkpoint_dir, "cl_state.pt")
    if os.path.exists(cl_state_path):
        try:
            cl_state = torch.load(cl_state_path, map_location='cpu')
            result['cl_state'] = cl_state
            result['cl_state_loaded'] = True
            print(f" Loaded continual learning state")
        except Exception as e:
            print(f"Failed to load cl_state: {e}")
    
    print(f"{'='*60}")
    print(f"✓ Checkpoint loaded successfully!")
    print(f"{'='*60}\n")
    
    return result


def _load_prototype_lib_legacy(proto_lib, state_dict: Dict, device: torch.device):

    import torch.nn as nn
    
    proto_lib.proto_dim = state_dict.get('proto_dim', proto_lib.proto_dim)
    proto_lib.threshold = state_dict.get('threshold', proto_lib.threshold)
    proto_lib.momentum = state_dict.get('momentum', proto_lib.momentum)
    proto_lib.max_prototypes = state_dict.get('max_prototypes', proto_lib.max_prototypes)
    proto_lib.total_samples = state_dict.get('total_samples', 0)
    
    proto_tensor = state_dict.get('prototypes', None)
    if proto_tensor is not None and proto_tensor.numel() > 0:
        num_protos = proto_tensor.shape[0]
        new_prototypes = []
        for i in range(num_protos):
            new_prototypes.append(nn.Parameter(proto_tensor[i].to(device)))
        proto_lib.prototypes = nn.ParameterList(new_prototypes)
    else:
        proto_lib.prototypes = nn.ParameterList()
    
    proto_lib.proto_counts = state_dict.get('proto_counts', [1] * len(proto_lib.prototypes))
    if len(proto_lib.proto_counts) != len(proto_lib.prototypes):
        proto_lib.proto_counts = [1] * len(proto_lib.prototypes)
    
    proto_lib.creation_history = state_dict.get('creation_history', list(range(len(proto_lib.prototypes))))



def integrate_protostream_save(trainer, training_args):
    save_protostream_checkpoint(
        output_dir=training_args.output_dir,
        model=trainer.model,
        rank=training_args.local_rank
    )


def integrate_protostream_load(model, training_args, device=None):

    if training_args.current_task_id > 0 and training_args.pretrained_checkpoint_path:
        return load_protostream_checkpoint(
            checkpoint_dir=training_args.pretrained_checkpoint_path,
            model=model,
            device=device or training_args.device
        )
    return None