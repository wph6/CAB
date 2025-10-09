CUDA_VISIBLE_DEVICES=0 python main_vit2vim.py \
    --model dual_vit2vim_deit_tiny_patch16 \
    --epochs 3000 --batch-size 64 --drop-path 0.0 --weight-decay 0.05 --num_workers 25 \
    --finetune deit_tiny_patch16_224 \
    --lr 5e-4 --mixup 0.8 --cutmix 1.0  \
    --data-path imagenet_subset_001 \
    --dist-eval --eval-interval 100 --no_amp --mode softdistill   \

CUDA_VISIBLE_DEVICES=1 python main_vit2vim.py \
    --model dual_vit2vim_deit_tiny_patch16 \
    --epochs 600 --batch-size 64 --drop-path 0.0 --weight-decay 0.05 --num_workers 25 \
    --finetune deit_tiny_patch16_224 \
    --lr 5e-4 --mixup 0.8 --cutmix 1.0  \
    --data-path imagenet_subset_005 \
    --dist-eval --eval-interval 20 --no_amp --mode softdistill   \

CUDA_VISIBLE_DEVICES=2 python main_vit2vim.py \
    --model dual_vit2vim_deit_tiny_patch16 \
    --epochs 300 --batch-size 64 --drop-path 0.0 --weight-decay 0.05 --num_workers 25 \
    --finetune deit_tiny_patch16_224 \
    --lr 5e-4 --mixup 0.8 --cutmix 1.0  \
    --data-path imagenet_subset_010 \
    --dist-eval --eval-interval 10 --no_amp --mode softdistill   \

CUDA_VISIBLE_DEVICES=3 python main_vit2vim.py \
    --model dual_vit2vim_deit_tiny_patch16 \
    --epochs 150 --batch-size 64 --drop-path 0.0 --weight-decay 0.05 --num_workers 25 \
    --finetune deit_tiny_patch16_224 \
    --lr 5e-4 --mixup 0.8 --cutmix 1.0  \
    --data-path imagenet_subset_020 \
    --dist-eval --eval-interval 10 --no_amp --mode softdistill   \
