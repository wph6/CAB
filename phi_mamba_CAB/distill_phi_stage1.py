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

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def main():
    parser = HfArgumentParser(SFTDistillConfig)
    (training_args,) = parser.parse_args_into_dataclasses()

    current_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
    experiment_name = f"stage1_phi15_lr{training_args.learning_rate}_{current_time}"
    training_args.output_dir = os.path.join("outputs", experiment_name)

    if "swanlab" in training_args.report_to and is_main_process():
        swanlab.init(project=f'NLPAttentransfer_stage1',name=experiment_name)

    dataset = load_from_disk(training_args.dataset_path)

    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    teacher_model = PhiForCausalLM.from_pretrained(
        "microsoft/phi-1_5", attn_implementation="eager"
    ).to(device)
    tokenizer.pad_token = tokenizer.eos_token 

    teacher_model.eval()
    teacher_model.requires_grad_(False)

    model_config = Config.from_json("assets/sample_config.json")
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