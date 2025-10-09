import os
import warnings
from copy import deepcopy
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from accelerate.utils import is_deepspeed_available
from transformers import AutoModelForCausalLM, PreTrainedModel, is_wandb_available

from trl.models import PreTrainedModelWrapper
# from trl.trainer.sft_trainer import SFTTrainer
from trl import SFTTrainer, SFTConfig
from train_configs import SFTDistillConfig
from torchtune.modules.loss import ForwardKLLoss
from torch.nn import KLDivLoss
from torch.cuda.amp import autocast
if is_deepspeed_available():
    import deepspeed



def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model

class DistillAligner(nn.Module):
    def __init__(self, student_dim, teacher_dim,num_layers):
        super().__init__()

        def make_proj_list():
            return nn.ModuleList([
                 nn.Sequential(
                    nn.Linear(student_dim, teacher_dim),
                    nn.SiLU(),
                    nn.Linear(teacher_dim, teacher_dim),
                 )
                for _ in range(num_layers)
            ])

        self.proj_B   = make_proj_list()
        self.proj_C   = make_proj_list()


    def flatten_teacher(self, t):
        """
        t: (B, n_heads, L, dim)
        return: (B, L, n_heads * dim)
        """
        B, H, L, D = t.shape
        return t.permute(0, 2, 1, 3).reshape(B, L, H * D)

    def flatten(self, t):
        """
        t: (B, n_heads, L, dim)
        return: (B, L, n_heads * dim)
        """
        B, L, H, D = t.shape
        return t.reshape(B, L, H * D)

    def forward(self, teacher_qk, student_bc):
        """
        teacher_qk: List of (q, k) tuples from teacher, each q/k: (B, n_heads, L, dim)
        student_bc: List of dicts with 'B', 'C' tensors: each: (B, 1, student_hdim, L)
        """
        loss = 0
        for i, ((q, k),(B,C)) in enumerate(zip(teacher_qk, student_bc)):
            q = self.flatten_teacher(q) 
            k = self.flatten_teacher(k)
            B = self.flatten(B) 
            C = self.flatten(C)

            B_proj = self.proj_B[i](B)
            C_proj = self.proj_C[i](C)

            loss += F.mse_loss(B_proj, k)
            loss += F.mse_loss(C_proj, q)

        return loss / len(teacher_qk)
    
class KDTrainer(SFTTrainer):
    _tag_names = ["trl", "kd"]

    def __init__(
        self,
        teacher_model,
        mode = 'kl',
        args: Optional[SFTDistillConfig] = None,
        stage = None,
        *sft_args,
        **kwargs,
    ):

        super().__init__(*sft_args, args=args, **kwargs)

        if self.is_deepspeed_enabled:
            self.teacher_model = self._prepare_deepspeed(teacher_model)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.kl_weight = args.kl_weight
        self.mode = mode
        self.stage = stage
        self.loss_fn = ForwardKLLoss()
        student_dim = self.model.config.Block1.core_input.d_state * self.model.config.Block1.core_input.n_qk_heads
        self.aligner = DistillAligner(student_dim=student_dim, teacher_dim=unwrap_model(self.teacher_model).config.hidden_size,num_layers=self.model.config.Block1.n_layers).to(teacher_model.device) 


    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        if not self.model.training:
            student_outputs = model(
                input_ids=inputs["input_ids"],
                labels=inputs["input_ids"]
            )
            loss = student_outputs["loss"]

            return (loss, student_outputs) if return_outputs else loss

        # ---------- training mode ----------

        use_distill = self.mode in ['distill','distillonly']
        self.teacher_model.eval()

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                output_QK=use_distill,
                use_cache=False,
            ) 

        student_outputs = model(
            input_ids=inputs["input_ids"],
            return_BC = use_distill,
            labels = inputs["input_ids"]
        )

        ce_loss = student_outputs["loss"]
        logs = {"ce_loss": ce_loss.item()}


        if self.mode  in ['distill','distillonly']:
            teacher_qk  = teacher_outputs.all_attn_matrices
            student_bc  = student_outputs['all_BC']
            
            teacher_layer_num = len(teacher_qk)
            student_layer_num = len(student_bc)
            assert teacher_layer_num % student_layer_num == 0
            layers_per_block = teacher_layer_num // student_layer_num 
            new_teacher_qk = [teacher_qk[i * layers_per_block] for i in range(student_layer_num)]

            with autocast(dtype=torch.bfloat16):
                distill_loss = self.aligner(new_teacher_qk,student_bc)
            
            logs["distill_loss"] = distill_loss.item()

        teacher_logits = teacher_outputs["logits"]
        student_logits = student_outputs["logits"]
        
        
        kl_loss = self.loss_fn(student_logits, teacher_logits,labels=inputs["input_ids"])
        logs["kl_loss"] = kl_loss.item()

        if self.mode == 'ce':
            loss = ce_loss
        elif self.mode == 'distillonly':
            loss = distill_loss
        elif self.mode == 'distill':
            loss = ce_loss + self.kl_weight * kl_loss + (1 - self.kl_weight) * distill_loss
        elif "kl" in self.mode:
            loss = ce_loss + kl_loss
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        logs = {
                "ce_loss": ce_loss.item(),
                "kl_loss": kl_loss.item(),
                "loss": loss.item()
                }
        logs["loss"] = loss.item()
        self.log(logs)

        return loss

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        

        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
