#stage1
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml distill_gpt_stage1.py --mode ourskl --max_seq_length 1024 --learning_rate 2e-5 --per_device_train_batch_size 32
#stage2
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml distill_gpt_stage2.py --mode ourskl --max_seq_length 1024 --learning_rate 2e-5 --per_device_train_batch_size 32

