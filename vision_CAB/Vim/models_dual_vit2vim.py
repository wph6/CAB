# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import trunc_normal_
from Vim.models_mamba import VisionMamba

def move_token_to_middle(q: torch.Tensor) -> torch.Tensor:
    cls = q[:, 0:1, :]
    rest = q[:, 1:, :]
    mid = rest.shape[1] // 2
    return torch.cat([rest[:, :mid], cls, rest[:, mid:]], dim=1)



class KernelMapper(nn.Module):
    def __init__(self, kernel_type="none"):
        super().__init__()
        self.kernel_type = kernel_type

    def forward_q(self, x):
        if self.kernel_type == "softmax":
            return F.softmax(x, dim=-1)  # Q: row-wise softmax (across features)
        elif self.kernel_type == "relu":
            return F.relu(x)
        elif self.kernel_type == "exp":
            return torch.exp(x)
        else:  # "none"
            return x

    def forward_k(self, x):
        if self.kernel_type == "softmax":
            return F.softmax(x, dim=-2)  # K: column-wise softmax (across tokens)
        elif self.kernel_type == "relu":
            return F.relu(x)
        elif self.kernel_type == "exp":
            return torch.exp(x)
        else:  # "none"
            return x


class DistillAligner_biv2(nn.Module):
    def __init__(self, student_hdim, teacher_hdim, n_heads,num_layers, kernel_type="none",projector_dim=128):
        super().__init__()
        self.n_heads = n_heads
        self.teacher_hdim = teacher_hdim
        out_dim = n_heads * teacher_hdim

        def make_proj_list():
             return nn.ModuleList([
                nn.Sequential(
                    nn.Linear(student_hdim, projector_dim),
                    nn.SiLU(),
                    nn.Linear(projector_dim, out_dim)
                ) for _ in range(num_layers)
            ])

        self.proj_B   = make_proj_list()
        self.proj_B_b = make_proj_list()
        self.proj_C   = make_proj_list()
        self.proj_C_b = make_proj_list()

        # kernel mapper for teacher features
        self.kernel = KernelMapper(kernel_type)


    def forward_single(self, x, proj):
        """
        x: (B, 1, h_dim, L)
        returns: (B, n_heads, L, dim)
        """
        x = x.squeeze(1).permute(0, 2, 1)           # (B, L, h_dim)
        x = proj(x)                                 # (B, L, n_heads * dim)
        return x

    def flatten_teacher(self, t):
        """
        t: (B, n_heads, L, dim)
        return: (B, L, n_heads * dim)
        """
        B, H, L, D = t.shape
        return t.permute(0, 2, 1, 3).reshape(B, L, H * D)

    def reverse_seq(self, x):
        """
        Reverse the token sequence along L dimension.
        x: (B, L, D)
        """
        return x.flip(dims=[1])

    def forward(self, teacher_qk, student_bc):
        loss = 0
        for i, ((q, k), act) in enumerate(zip(teacher_qk, student_bc)):
            # student projections
            B   = self.forward_single(act['B'],   self.proj_B[i])
            B_b = self.forward_single(act['B_b'], self.proj_B_b[i])
            C   = self.forward_single(act['C'],   self.proj_C[i])
            C_b = self.forward_single(act['C_b'], self.proj_C_b[i])

            # teacher kernels
            q = self.flatten_teacher(q)           # (B, L, D)
            k = self.flatten_teacher(k)           # (B, L, D)

            q_ker = self.kernel.forward_q(q)
            k_ker = self.kernel.forward_k(k)

            q_ker = move_token_to_middle(q_ker)
            k_ker = move_token_to_middle(k_ker)

            # reversed kernels
            q_ker_rev = self.reverse_seq(q_ker)
            k_ker_rev = self.reverse_seq(k_ker)

            # losses
            loss += F.mse_loss(B,   k_ker)
            loss += F.mse_loss(B_b, k_ker_rev)
            loss += F.mse_loss(C,   q_ker)
            loss += F.mse_loss(C_b, q_ker_rev)

        return loss / len(teacher_qk)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, mode=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.mode = mode


    def forward(self, x, teacher_act=None, return_act=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if self.mode == 'copy' and teacher_act is not None:
            attn = teacher_act.softmax(dim=-1)
        elif self.mode == 'copy_q' and teacher_act is not None:
            teacher_q = teacher_act
            attn_logits = (teacher_q @ k.transpose(-2, -1)) * self.scale
            attn = attn_logits.softmax(dim=-1)
        elif self.mode == 'copy_k' and teacher_act is not None:
            teacher_k = teacher_act
            attn_logits = (q @ teacher_k.transpose(-2, -1)) * self.scale
            attn = attn_logits.softmax(dim=-1)
        else:
            attn_logits = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn_logits.softmax(dim=-1)
        if self.mode == 'copy_v' and teacher_act is not None:
            teacher_v = teacher_act
            x = (attn @ teacher_v).transpose(1, 2).reshape(B, N, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if return_act == 'attention':
            return x, attn_logits
        elif return_act is not None and 'qk' in return_act:
            return x, (q, k)
        elif return_act == 'q':
            return x, q
        elif return_act == 'k':
            return x, k
        elif return_act == 'v':
            return x, v
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, mode=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, mode=mode)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=0)

    def forward(self, x, teacher_act=None, return_act=None):

        attn_outputs = self.attn(self.norm1(x), teacher_act=teacher_act, return_act=return_act)
        if return_act is not None:
            attn_result, act = attn_outputs
        else:
            attn_result = attn_outputs

        x = x + attn_result
        x = x + self.mlp(self.norm2(x))
        if return_act is not None:
            return x, act
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 norm_layer=nn.LayerNorm, mode=None, global_pool=False,):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.mode = mode

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_rate)

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, norm_layer=norm_layer, mode=mode)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, teacher_act=None, return_act=None):
        if teacher_act is not None:
            assert 'copy' in self.mode
            # pad if we copy fewer blocks
            if len(teacher_act) < len(self.blocks):
                teacher_act = teacher_act + [None] * (len(self.blocks) - len(teacher_act))
        else:
            teacher_act = [None] * len(self.blocks)

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        attns = []
        for i, blk in enumerate(self.blocks):
            if return_act is not None:
                x, act = blk(x, return_act=return_act)
                attns.append(act)
            else:
                x = blk(x, teacher_act=teacher_act[i])

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        if return_act is not None:
            return outcome, attns
        return outcome


    def forward(self, x, teacher_act=None, return_act=None):
        x = self.forward_features(x, teacher_act=teacher_act, return_act=return_act)
        if return_act is not None:
            x, act = x
            return self.head(x), act
        else:
            return self.head(x)

    def no_weight_decay(self):
        return {'student.' + k for k in self.student.no_weight_decay()}
    
class DualViT2Vim(nn.Module):
    """
    Vision Transformer with support for global average pooling
    Has two streams (one is a teacher, the other is a student)
    """
    def __init__(self, mode='distill', drop_path_rate=0,
                 teacher_kwargs=None, student_kwargs=None, end_layer=-3,kernel_type = 'none'):
        super().__init__()
        assert mode in {'softdistill','copy', 'copy_q', 'copy_k', 'copy_v', 'distill', 'distill_q', 'distill_k', 'distill_v','soft'}
        self.mode = mode
        self.drop_path_rate = drop_path_rate
        self.teacher_depth = teacher_kwargs['depth']
        self.student_depth = student_kwargs['depth']
        self.teacher = VisionTransformer(mode='teacher', **teacher_kwargs)
        self.student = VisionMamba(mode=mode, **student_kwargs)
        self.dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.teacher_depth)]

        self.aligner = DistillAligner_biv2(
                    teacher_hdim=teacher_kwargs['embed_dim']//teacher_kwargs['num_heads'], 
                    student_hdim=16, # d_state
                    n_heads=teacher_kwargs['num_heads'],
                    num_layers=student_kwargs['depth'],
                    kernel_type= kernel_type
                )

    def forward(self, x):
        # attention activation to get from the teacher
        if self.mode in {'soft'}:
            return_act = None
        elif self.mode in {'softdistill'}:
            return_act = 'qkBC'
        elif self.mode in {'copy'}:
            return_act = 'attention'
        elif self.mode in {'copy_q', 'distill_q'}:
            return_act = 'q'
        elif self.mode in {'copy_k', 'distill_k'}:
            return_act = 'k'
        elif self.mode in {'copy_v', 'distill_v'}:
            return_act = 'v'
        else:
            raise NotImplementedError
        with torch.no_grad():
            # if self.training or 'copy' in self.mode:
            if self.mode != 'soft' and (self.training or 'copy' in self.mode):
                teacher_out, teacher_act = self.teacher.forward(x,return_act=return_act)
                if 'qk' in return_act:
                    teacher_act = [(q.detach(), k.detach()) for (q, k) in teacher_act]
                else:
                    teacher_act = [act.detach() for act in teacher_act]
            elif self.mode == 'soft' and self.training:
                with torch.no_grad():
                    teacher_out = self.teacher(x)
        # forward student
        if 'copy' in self.mode:
            # teacher_act to copy
            act_to_copy = teacher_act[:self.teacher_depth + self.end_layer]
            return self.student(x,teacher_act=act_to_copy)
        
        elif self.mode == 'softdistill' and self.training:

            student_out, student_act = self.student(x,return_act=return_act)

            teacher_layer_num = len(teacher_act)
            student_layer_num = len(student_act)
            assert student_layer_num % teacher_layer_num == 0
            layers_per_block = student_layer_num // teacher_layer_num 
            new_teacher_atts = [teacher_act[i // layers_per_block] for i in range(student_layer_num)]

            distill_loss = self.aligner(new_teacher_atts, student_act)
            
            T = 1
            soft_loss = F.kl_div(
                F.log_softmax(student_out / T, dim=-1),
                F.softmax(teacher_out / T, dim=-1),
                reduction='batchmean'
            ) * (T * T)


            return student_out, soft_loss, distill_loss

        elif self.mode == 'soft' and self.training:
            student_out = self.student(x)
            T = 1
            distill_loss = F.kl_div(
                F.log_softmax(student_out / T, dim=-1),
                F.softmax(teacher_out / T, dim=-1),
                reduction='batchmean'
            ) * (T * T)
            return student_out, distill_loss

        else:
            return self.student(x)


    def no_weight_decay(self):
        return {'student.' + k for k in self.student.no_weight_decay()}


def dual_vit2vim_deit_tiny_patch16(mode='distill', drop_path_rate=0, end_layer=-3,kernel_type='none', **kwargs):
    t_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    s_kwargs = dict(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, 
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)

    model = DualViT2Vim(mode=mode, drop_path_rate=drop_path_rate, end_layer=end_layer,kernel_type= kernel_type,
                        teacher_kwargs=t_kwargs, student_kwargs=s_kwargs)
    return model

def dual_vit2vim_deit_small_patch16(mode='distill', drop_path_rate=0, end_layer=-3,kernel_type='none', **kwargs):
    t_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    s_kwargs = dict(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, 
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)

    model = DualViT2Vim(mode=mode, drop_path_rate=drop_path_rate, end_layer=end_layer,kernel_type= kernel_type,
                        teacher_kwargs=t_kwargs, student_kwargs=s_kwargs)
    return model

def dual_vit2vim_deit_base_patch16(mode='distill', drop_path_rate=0, end_layer=-3,kernel_type='none', **kwargs):
    t_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    s_kwargs = dict(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, 
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)

    model = DualViT2Vim(mode=mode, drop_path_rate=drop_path_rate, end_layer=end_layer,kernel_type= kernel_type,
                        teacher_kwargs=t_kwargs, student_kwargs=s_kwargs)
    return model

def dual_vit2vim_large_patch16(mode='distill', drop_path_rate=0, end_layer=-3,kernel_type='none', **kwargs):
    t_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    s_kwargs = dict(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, 
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)

    model = DualViT2Vim(mode=mode, drop_path_rate=drop_path_rate, end_layer=end_layer,kernel_type= kernel_type,
                                  teacher_kwargs=t_kwargs, student_kwargs=s_kwargs)
    return model
