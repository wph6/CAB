import os
import sys
import transformers
import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional, NewType
from transformers import HfArgumentParser
from trl import SFTConfig

DataClassType = NewType("DataClassType", Any)

@dataclass
class SFTDistillConfig(SFTConfig):
    """
    Arguments related to the distillation process.
    """

    prev_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the previous distilled model in this progressive distillation."},
    )
    kl_weight: float = field(
        default=0.5,
        metadata={"help": "Ratio of KL loss."},
    )
    # ce_weight: float = field(
    #     default=1,
    #     metadata={"help": "Ratio of CE loss."},
    # )
    mode: str = field(
        default= 'initkl',
        metadata={"help": "Training mode."},
    )
    train_datasets_path: List[str] = field(
        default=None,
        metadata={"help": "Training datasets."},
    )
    test_datasets_path: List[str] = field(
        default=None,
        metadata={"help": "Test datasets."},
    )
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    dataset_batch_size: int = 4
    num_train_epochs: int = 1
    report_to: str = 'swanlab'
    logging_steps:int = 200
    logging_strategy: str = 'steps'
    learning_rate: float = 5e-4
    save_only_model:bool = True
    save_steps: int = 10000
    eval_steps:int =  3200
    evaluation_strategy: str = "no"

    warmup_ratio: float = field(
        default=0.1, 
        metadata={"help": "Ratio of warmup steps over total training steps."},
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Enable 16-bit (mixed) precision training via AMP."},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Enable bfloat16 mixed-precision training (A100/H100 only)."},
    )