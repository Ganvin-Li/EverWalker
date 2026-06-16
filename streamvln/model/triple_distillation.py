
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class TripleDistillationLoss(nn.Module):

    def __init__(
        self,
        model,
        lambda_sp: float = 1.0,
        lambda_pp: float = 0.5,
        lambda_cp: float = 0.3,
        temperature: float = 2.0,
        distill_layers: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.model = model
        self.lambda_sp = lambda_sp
        self.lambda_pp = lambda_pp
        self.lambda_cp = lambda_cp
        self.temperature = temperature
        self.distill_layers = distill_layers
        
        self.teacher_outputs = {}
        self.teacher_prototypes = []
        
        self.distillation_enabled = False
    
    def save_teacher_state(self):

        if hasattr(self.model, 'prototype_lib'):
            self.model.prototype_lib.freeze_current_prototypes()
            
            current_protos = [p.detach().clone() for p in self.model.prototype_lib.prototypes]
            self.teacher_prototypes.append(current_protos)
        
        print(f"[Distillation] Saved teacher state with {len(current_protos)} prototypes")
        self.distillation_enabled = True
    
    def compute_single_prototype_distillation(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor
    ) -> torch.Tensor:

        student_log_probs = F.log_softmax(student_output / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_output / self.temperature, dim=-1)
        
        kl_div = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        )
        
        loss_sp = kl_div * (self.temperature ** 2)
        
        return loss_sp
    
    def compute_prototype_pair_distillation(
        self,
        model,
        proto_idx1: int,
        proto_idx2: int,
        sample_input: Dict
    ) -> torch.Tensor:

        if not self.teacher_prototypes:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        last_teacher_protos = self.teacher_prototypes[-1]
        if proto_idx1 >= len(last_teacher_protos) or proto_idx2 >= len(last_teacher_protos):
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        teacher_proto1 = last_teacher_protos[proto_idx1]
        teacher_proto2 = last_teacher_protos[proto_idx2]
        
        if proto_idx1 >= len(model.prototype_lib) or proto_idx2 >= len(model.prototype_lib):
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        student_proto1 = model.prototype_lib.prototypes[proto_idx1]
        student_proto2 = model.prototype_lib.prototypes[proto_idx2]
        
        alpha = 0.5
        teacher_combo = alpha * teacher_proto1 + (1 - alpha) * teacher_proto2
        student_combo = alpha * student_proto1 + (1 - alpha) * student_proto2
        
        teacher_combo = F.normalize(teacher_combo, p=2, dim=-1)
        student_combo = F.normalize(student_combo, p=2, dim=-1)
        
        similarity = F.cosine_similarity(teacher_combo, student_combo, dim=-1)
        loss_pp = 1.0 - similarity.mean()
        
        return loss_pp
    
    def compute_cross_prototype_distillation(
        self,
        model,
        current_proto_idx: int,
        sample_input: Dict
    ) -> torch.Tensor:

        if not self.teacher_prototypes:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        total_loss = 0.0
        count = 0
        
        for task_id, teacher_protos in enumerate(self.teacher_prototypes):
            for teacher_proto_idx, teacher_proto in enumerate(teacher_protos):
                if teacher_proto_idx == current_proto_idx and task_id == len(self.teacher_prototypes) - 1:
                    continue
                
                if teacher_proto_idx < len(model.prototype_lib):
                    student_proto = model.prototype_lib.prototypes[teacher_proto_idx]
                    
                    drift = F.mse_loss(student_proto, teacher_proto.to(student_proto.device))
                    total_loss += drift
                    count += 1
        
        if count > 0:
            loss_cp = total_loss / count
        else:
            loss_cp = torch.tensor(0.0, device=next(model.parameters()).device)
        
        return loss_cp
    
    def compute_distillation_loss(
        self,
        model,
        batch: Dict,
        student_outputs: Optional[torch.Tensor] = None,
        teacher_outputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.distillation_enabled:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        device = next(model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        
        if self.lambda_sp > 0 and student_outputs is not None and teacher_outputs is not None:
            loss_sp = self.compute_single_prototype_distillation(student_outputs, teacher_outputs)
            total_loss += self.lambda_sp * loss_sp
        
        if self.lambda_pp > 0 and len(model.prototype_lib) >= 2:
            num_protos = len(model.prototype_lib)
            idx1 = torch.randint(0, num_protos, (1,)).item()
            idx2 = torch.randint(0, num_protos, (1,)).item()
            if idx1 != idx2:
                loss_pp = self.compute_prototype_pair_distillation(model, idx1, idx2, batch)
                total_loss += self.lambda_pp * loss_pp
        
        if self.lambda_cp > 0 and len(model.prototype_lib) > 0:
            if hasattr(model, 'current_proto_idx') and model.current_proto_idx is not None:
                current_idx = model.current_proto_idx
            else:
                current_idx = 0
            
            loss_cp = self.compute_cross_prototype_distillation(model, current_idx, batch)
            total_loss += self.lambda_cp * loss_cp
        
        return total_loss
    
    def get_teacher_output(
        self,
        model,
        batch: Dict,
        proto_idx: int
    ) -> torch.Tensor:

        was_training = model.training
        model.eval()
        
        with torch.no_grad():
            if proto_idx < len(self.teacher_prototypes[-1]) if self.teacher_prototypes else False:
                original_proto = model.prototype_lib.prototypes[proto_idx].data.clone()
                model.prototype_lib.prototypes[proto_idx].data = self.teacher_prototypes[-1][proto_idx]
                
                outputs = model(**batch)
                
                model.prototype_lib.prototypes[proto_idx].data = original_proto
            else:
                outputs = model(**batch)
        
        if was_training:
            model.train()
        
        return outputs.logits if hasattr(outputs, 'logits') else outputs
    
    def __repr__(self):
        return f"TripleDistillationLoss(λ_sp={self.lambda_sp}, λ_pp={self.lambda_pp}, " \
               f"λ_cp={self.lambda_cp}, T={self.temperature})"