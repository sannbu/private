/main.py
+8
-6

import argparse
import csv
import os
import re
import time
import torch
import torchvision

from data import dataloader
from model import Pix2Pix


EXECUTION_ID = time.strftime('%m_%d_%H_%M_%S')

parser = argparse.ArgumentParser()


parser.add_argument('--data_root', required=True, type=str)
parser.add_argument('--data_resize', default=286, type=int)
parser.add_argument('--data_crop', default=256, type=int)
# Varsayılan olarak A/B klasörlü RGB->Thermal veri seti kullanılır
parser.add_argument('--dataset', default='rgb2thermal', type=str)
parser.add_argument('--data_invert', action='store_true')
parser.add_argument('--out_root', default=os.path.join('.', 'output'), type=str)

# Training
parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--val_num_batches', default=5, type=int)
parser.add_argument('--pretrain_timestamp', type=str)
parser.add_argument('--save_model_rate', default=1, type=int)

# Optimization
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--lambda_l1', default=100.0, type=float)
parser.add_argument('--lambda_d', default=0.5, type=float)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_3ch(t: torch.Tensor) -> torch.Tensor:
    """N,C,H,W tensörünü 3 kanala dönüştür (C==1 ise tekrarla)."""
    if t.dim() != 4:
        raise ValueError(f"Expected 4D tensor (N,C,H,W), got {t.shape}")
    if t.size(1) == 3:
@@ -81,51 +82,52 @@ if __name__ == '__main__':
        with open(os.path.join(out_root, 'config.txt'), 'w') as config:
            config.write(str(args))

        train_dataloader = dataloader(
            os.path.join(args.data_root, 'train'), args.dataset,
            invert=args.data_invert, train=True, shuffle=True,
            device=device, batch_size=args.batch_size, resize=args.data_resize, crop=args.data_crop
        )
        val_dataloader = dataloader(
            os.path.join(args.data_root, 'val'), args.dataset,
            invert=args.data_invert, train=False, shuffle=False,
            device=device, batch_size=args.batch_size, resize=args.data_resize, crop=args.data_crop
        )

    elif args.mode == 'test':
        out_root = os.path.join(args.out_root, args.pretrain_timestamp)
        out_image_path = os.path.join(out_root, 'images', f"test_{EXECUTION_ID}")
        ensure_dir(out_image_path)

        test_dataloader = dataloader(
            os.path.join(args.data_root, 'test'), args.dataset,
            invert=args.data_invert, train=False, shuffle=False,
            device=device, batch_size=args.batch_size, resize=args.data_resize, crop=args.data_crop
        )

    
    # Pix2Pix modeli varsayılan olarak 3->1 kanal dönüşümü yapar; ek dataset argümanı gerekmiyor
    model = Pix2Pix(lr=args.lr, lambda_l1=args.lambda_l1, lambda_d=args.lambda_d)

    start_epoch = 0
    if args.pretrain_timestamp:
        checkpoint_dir = os.path.join(args.out_root, args.pretrain_timestamp)
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        # En büyük epoch'u bul
        best_epoch = -1
        best_file = None
        for fname in os.listdir(checkpoint_dir):
            m = re.match(r'^epoch_(\d+)\.pt$', fname)
            if m:
                ep = int(m.group(1))
                if ep > best_epoch:
                    best_epoch = ep
                    best_file = fname

        if best_file is None:
            raise FileNotFoundError(f"No 'epoch_*.pt' checkpoint found in {checkpoint_dir}")

        print(f"Using pretrained model... {args.pretrain_timestamp}/{best_file}")
        checkpoint = torch.load(os.path.join(checkpoint_dir, best_file), map_location=device)
        model.load_state_dict(checkpoint['state'])
        start_epoch = int(checkpoint.get('epoch', best_epoch)) + 1
