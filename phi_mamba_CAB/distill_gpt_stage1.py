import os
import numpy as np

from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from modules.lm_head import LMHeadModel
from modules.modeling_phi import PhiForCausalLM
from modules.modeling_gpt2 import GPT2LMHeadModel
from utils.config import Config
from transformers import HfArgumentParser, DataCollatorForLanguageModeling

import datetime
from kd_trainer_gpt import KDTrainer
import wandb
from train_configs import SFTDistillConfig
import torch
import swanlab

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def main():
    parser = HfArgumentParser(SFTDistillConfig)
    (training_args,) = parser.parse_args_into_dataclasses()

    current_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
    experiment_name = f"stage1_gpt_lr{training_args.learning_rate}_{current_time}"
    training_args.output_dir = os.path.join("outputs", experiment_name)

    if "swanlab" in training_args.report_to and is_main_process():
        swanlab.init(project=f'NLPAttentransfer_stage1',name=experiment_name)

    
    dataset = load_from_disk(training_args.dataset_path)

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2", local_files_only=True)
    teacher_model = GPT2LMHeadModel.from_pretrained("distilbert/distilgpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token 

    teacher_model.eval()
    teacher_model.requires_grad_(False)

    model_config = Config.from_json("assets/m1_config.json")
    student_model = LMHeadModel(model_config,mode=training_args.mode).to(device)

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

    trainer.train()

if __name__ == "__main__":
    main()