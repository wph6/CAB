import os
import numpy as np
from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from modules.lm_head import LMHeadModel
from modules.modeling_gpt2 import GPT2LMHeadModel
from utils.config import Config
from transformers import HfArgumentParser, DataCollatorForLanguageModeling

import datetime
from kd_trainer_gpt import KDTrainer
import swanlab
from train_configs import SFTDistillConfig
import torch

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def main():
    parser = HfArgumentParser(SFTDistillConfig)
    (training_args,) = parser.parse_args_into_dataclasses()

    training_args.output_dir = f'trainer_output/{experiment_name}'

    current_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
    experiment_name = f"stage2_gpt_lr{training_args.learning_rate}_{current_time}"
    training_args.output_dir = os.path.join("outputs", experiment_name)

    if "swanlab" in training_args.report_to and is_main_process():
        swanlab.init(project=f'NLPAttentransfer',name=experiment_name)

    dataset = load_from_disk(training_args.dataset_path)

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2", local_files_only=True)
    teacher_model = GPT2LMHeadModel.from_pretrained("distilbert/distilgpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token 

    teacher_model.eval()
    teacher_model.requires_grad_(False)

    model_config = Config.from_json("assets/m1_config.json")
    student_model = LMHeadModel(model_config,mode=training_args.mode).to(device)

    if training_args.mode == "ourskl" and training_args.student_init_path:
        state_dict = torch.load(training_args.student_init_path, map_location="cpu")
        student_model.load_state_dict(state_dict)

    elif training_args.mode == "comparekl" and training_args.student_init_path:
        state_dict = torch.load(training_args.student_init_path, map_location="cpu")
        student_model.load_state_dict(state_dict)

    elif training_args.mode == 'initkl' :
        for layer_id in range(student_model.config.Block1.n_layers):
            teacher_layer = teacher_model.transformer.h[layer_id]
            mamba_layer = student_model.backbone.layers[layer_id]
            mamba_layer.input_layernorm.load_state_dict({
                "weight": teacher_layer.ln_1.weight.clone(),
                "bias": teacher_layer.ln_1.bias.clone(),
            })
            mamba_layer.mlp.fc1.weight.data.copy_(teacher_layer.mlp.c_fc.weight.T)
            mamba_layer.mlp.fc1.bias.data.copy_(teacher_layer.mlp.c_fc.bias)
            mamba_layer.mlp.fc2.weight.data.copy_(teacher_layer.mlp.c_proj.weight.T)
            mamba_layer.mlp.fc2.bias.data.copy_(teacher_layer.mlp.c_proj.bias)

            c_attn_weight = teacher_layer.attn.c_attn.weight.T
            c_attn_bias = teacher_layer.attn.c_attn.bias
            qkv_size = c_attn_weight.shape[0] // 3
            q_weight, k_weight, v_weight = torch.split(c_attn_weight, qkv_size)
            q_bias, k_bias, v_bias = torch.split(c_attn_bias, qkv_size)

            in_proj = mamba_layer.mixer.in_proj.weight 
            d_model = student_model.config.MixerModel.input.d_model
            d_state = student_model.config.Block1.core_input.d_state
            n_qk_heads = student_model.config.Block1.core_input.n_qk_heads
            head_dim = d_state * n_qk_heads

            in_proj.data[:d_model].copy_(v_weight.T)
            in_proj.data[d_model : d_model + head_dim].copy_(k_weight.T)
            in_proj.data[d_model + head_dim : d_model + 2 * head_dim].copy_(q_weight.T)
            if mamba_layer.mixer.in_proj.bias is not None:
                in_proj_bias = mamba_layer.mixer.in_proj.bias
                in_proj_bias.data[:d_model].copy_(v_bias)
                in_proj_bias.data[d_model : d_model + head_dim].copy_(k_bias)
                in_proj_bias.data[d_model + head_dim : d_model + 2 * head_dim].copy_(q_bias)

            mamba_layer.mixer.out_proj.weight.data.copy_(teacher_layer.attn.c_proj.weight.T)
            if mamba_layer.mixer.out_proj.bias is not None:
                mamba_layer.mixer.out_proj.bias.data.copy_(teacher_layer.attn.c_proj.bias)

    trainer = KDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        mode = training_args.mode,
    )

    trainer.train()


if __name__ == "__main__":
    main()