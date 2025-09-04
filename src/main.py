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

# Data
parser.add_argument('--data_root', required=True, type=str)
parser.add_argument('--data_resize', default=286, type=int)
parser.add_argument('--data_crop', default=256, type=int)
parser.add_argument('--dataset', required=True, type=str)
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
        return t
    if t.size(1) == 1:
        return t.repeat(1, 3, 1, 1)
    # Farklı kanal sayısı varsa ilk 3 kanalı al
    return t[:, :3, ...]


def save_triptych(x: torch.Tensor, y: torch.Tensor, g: torch.Tensor, out_path: str):
    """
    x: input (N, 3, H, W) veya (N, 1, H, W)
    y: target (N, 1, H, W) veya (N, 3, H, W)
    g: output (N, 1, H, W) veya (N, 3, H, W)
    Görsel: [x | y | g] tek satır.
    """
    x3 = to_3ch(x)
    y3 = to_3ch(y)
    g3 = to_3ch(g)
    grid = torch.cat([x3, y3, g3], dim=3)  # genişlik boyunca birleştir
    torchvision.utils.save_image(
        grid, out_path, nrow=1, normalize=True, range=(-1, 1)
    )


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.out_root)

    if args.mode == 'train':
        out_root = os.path.join(args.out_root, f"{EXECUTION_ID}")
        out_image_path = os.path.join(out_root, 'images')
        ensure_dir(out_image_path)

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

    model = Pix2Pix(lr=args.lr, lambda_l1=args.lambda_l1, lambda_d=args.lambda_d, dataset=args.dataset)

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

    model.to(device)

    if args.mode == 'train':
        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            train_loss_path = os.path.join(out_root, 'train_loss.csv')
            val_loss_path = os.path.join(out_root, 'val_loss.csv')
            # append modunda aç
            with open(train_loss_path, 'a', newline='') as train_loss_file, \
                 open(val_loss_path, 'a', newline='') as val_loss_file:

                train_writer = csv.writer(train_loss_file, delimiter=',')
                val_writer = csv.writer(val_loss_file, delimiter=',')

                print(f"\n------------ Epoch {epoch} ------------")
                model.scheduler_step()

                clock_tick = time.time()
                for batch_index, data in enumerate(train_dataloader):
                    loss, output_g = model.train(data)

                    if batch_index % 100 == 0:
                        stats_string = ''.join(f" | {k} = {v:6.3f}" for k, v in loss.items())
                        print(f"[TRAIN]  batch_index = {batch_index:03d}{stats_string}")
                        # CSV: epoch, batch_index, her bir metrik ayrı kolona
                        row = [epoch + 1, batch_index] + [v for _, v in loss.items()]
                        train_writer.writerow(row)

                        x, y = data  # (N,C,H,W)
                        save_triptych(x, y, output_g,
                                      os.path.join(out_image_path, f"train_{epoch}_{batch_index}.png"))

                clock_tok = time.time()
                print(f"[CLOCK] Time taken: {(clock_tok - clock_tick) / 60:.3f} minutes")

                for batch_index, data in enumerate(val_dataloader):
                    if batch_index >= args.val_num_batches:
                        break
                    loss, output_g = model.eval(data)
                    stats_string = ''.join(f" | {k} = {v:6.3f}" for k, v in loss.items())
                    print(f"[VAL]    batch_index = {batch_index:03d}{stats_string}")
                    row = [epoch + 1, batch_index] + [v for _, v in loss.items()]
                    val_writer.writerow(row)

                    x, y = data
                    save_triptych(x, y, output_g,
                                  os.path.join(out_image_path, f"val_{epoch}_{batch_index}.png"))

            if epoch % args.save_model_rate == 0:
                checkpoint_file_path = os.path.join(out_root, f"epoch_{epoch}.pt")
                torch.save({
                    'state': model.state_dict(),
                    'epoch': epoch,
                }, checkpoint_file_path)

    elif args.mode == 'test':
        for batch_index, data in enumerate(test_dataloader):
            loss, output_g = model.eval(data)

            # Aralıktaki görsel isimlemesi (başlangıç-bitiş)
            batch_start = batch_index * args.batch_size
            # Gerçek batch boyutu (son batch küçük olabilir)
            curr_bs = data[0].size(0)
            batch_end = batch_start + curr_bs - 1

            x, y = data
            save_triptych(x, y, output_g,
                          os.path.join(out_image_path, f"{batch_start}_{batch_end}.png"))
