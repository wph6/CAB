import os
import shutil
import random
from pathlib import Path

def create_imagenet_subset(src_dir, dst_dir, ratio=0.1, seed=42):
    random.seed(seed)
    
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    train_src = src_dir / 'train'
    val_src = src_dir / 'val'

    train_dst = dst_dir / 'train'
    val_dst = dst_dir / 'val'

    # Create target directories
    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)

    # Process training set
    for class_dir in sorted(train_src.iterdir()):
        if not class_dir.is_dir():
            continue
        images = sorted(list(class_dir.glob('*.JPEG')))
        sampled_images = random.sample(images, max(1, int(len(images) * ratio)))

        target_class_dir = train_dst / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sampled_images:
            shutil.copy(img_path, target_class_dir / img_path.name)

    print(f" Train subset created at: {train_dst}")

    # Copy the entire validation set
    if not (val_dst.exists() and any(val_dst.iterdir())):
        shutil.copytree(val_src, val_dst, dirs_exist_ok=True)
        print(f" Val copied to: {val_dst}")
    else:
        print(f" Val folder already exists at: {val_dst}, skipping copy.")

# Example usage
create_imagenet_subset(
    src_dir='/mnt/share/imagenet',     
    dst_dir='/mnt/share/imagenet_subset',  
    ratio=0.1,              
    seed=2025  
)
