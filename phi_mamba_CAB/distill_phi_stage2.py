import os
import numpy as np
from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from modules.lm_head import LMHeadModel
from modules.modeling_phi import PhiForCausalLM
from utils.config import Config
from transformers import HfArgumentParser, DataCollatorForLanguageModeling

import datetime
from kd_trainer import KDTrainer
import swanlab
from utils.train_configs import SFTDistillConfig
import torch
from safetensors.torch import load_file

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def main():
    parser = HfArgumentParser(SFTDistillConfig)
    (training_args,) = parser.parse_args_into_dataclasses()

    current_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
    experiment_name = f"stage2_phi15_lr{training_args.learning_rate}_{current_time}"
    training_args.output_dir = os.path.join("outputs", experiment_name)

    if "swanlab" in training_args.report_to and is_main_process():
        swanlab.init(project=f'NLPAttentransfer',name=experiment_name)

    dataset = load_from_disk(training_args.dataset_path)

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    teacher_model = PhiForCausalLM.from_pretrained(
        "microsoft/phi-1_5", attn_implementation="eager").to(device)
    tokenizer.pad_token = tokenizer.eos_token 

    teacher_model.eval()
    teacher_model.requires_grad_(False)

    model_config = Config.from_json("assets/sample_config.json")
    student_model = LMHeadModel(model_config,mode=training_args.mode).to(device)

    if training_args.mode == "ourskl" and training_args.student_init_path:
        state_dict = torch.load(training_args.student_init_path, map_location="cpu")
        student_model.load_state_dict(state_dict)

    elif training_args.mode == "comparekl" and training_args.student_init_path:
        state_dict = torch.load(training_args.student_init_path, map_location="cpu")
        student_model.load_state_dict(state_dict)

    elif training_args.mode == 'initkl':
        teacher_state_dict = load_file(
            "./models/phi-1_5/model.safetensors",
            device="cpu"
        )

        for layer_id in range(student_model.config.Block1.n_layers):
            prefix = f"model.layers.{layer_id}."
            mamba_layer = student_model.backbone.layers[layer_id]

            mamba_layer.input_layernorm.weight.data.copy_(teacher_state_dict[prefix + "input_layernorm.weight"])
            mamba_layer.input_layernorm.bias.data.copy_(teacher_state_dict[prefix + "input_layernorm.bias"])

            mamba_layer.mlp.fc1.weight.data.copy_(teacher_state_dict[prefix + "mlp.fc1.weight"])
            mamba_layer.mlp.fc1.bias.data.copy_(teacher_state_dict[prefix + "mlp.fc1.bias"])
            mamba_layer.mlp.fc2.weight.data.copy_(teacher_state_dict[prefix + "mlp.fc2.weight"])
            mamba_layer.mlp.fc2.bias.data.copy_(teacher_state_dict[prefix + "mlp.fc2.bias"])
                    
            q_weight = teacher_state_dict[prefix + "self_attn.q_proj.weight"]
            k_weight = teacher_state_dict[prefix + "self_attn.k_proj.weight"]
            v_weight = teacher_state_dict[prefix + "self_attn.v_proj.weight"]

            in_proj = mamba_layer.mixer.in_proj.weight 
            d_model = student_model.config.MixerModel.input.d_model
            d_state = student_model.config.Block1.core_input.d_state
            n_qk_heads = student_model.config.Block1.core_input.n_qk_heads
            head_dim = d_state * n_qk_heads

            in_proj.data[:d_model].copy_(v_weight)
            in_proj.data[d_model : d_model + head_dim].copy_(k_weight)
            in_proj.data[d_model + head_dim : d_model + 2 * head_dim].copy_(q_weight)

            mamba_layer.mixer.out_proj.weight.data.copy_(teacher_state_dict[prefix + "self_attn.dense.weight"])
            if mamba_layer.mixer.out_proj.bias is not None:
                mamba_layer.mixer.out_proj.bias.data.copy_(teacher_state_dict[prefix + "self_attn.dense.bias"])

    trainer = KDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset= dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        mode = training_args.mode,
    )

    if training_args.prev_checkpoint_path:
        trainer.train(resume_from_checkpoint=training_args.prev_checkpoint_path)
    else:
        trainer.train()
if __name__ == "__main__":
    main()