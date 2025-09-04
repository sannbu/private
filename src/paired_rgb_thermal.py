import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode as IM

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _to_tensor_01(img: Image.Image) -> torch.Tensor:
    """
    PIL -> torch float tensor in [0,1], CxHxW. 8/16-bit griyi güvenle ele alır.
    """
    if img.mode in ("I;16", "I;16B", "I;16L"):
        arr = np.array(img, dtype=np.uint16).astype(np.float32)
        arr = arr / 65535.0
        if arr.ndim == 2:
            arr = arr[None, ...]  # 1xHxW
        else:
            arr = arr.transpose(2, 0, 1)  # CxHxW
        return torch.from_numpy(arr)
    else:
        return TF.to_tensor(img)  # [0,1], CxHxW


def _normalize_tanh(x: torch.Tensor) -> torch.Tensor:
    # [0,1] -> [-1,1]
    return x * 2.0 - 1.0


class PairedFoldersDataset(Dataset):
    """
    A = RGB (3 kanal), B = Termal GT (1 kanal) klasör eşlemeli veri seti.
    Eşleştirme: dosya 'stem' (isim) bazlı. Örn: 0001.jpg <-> 0001.png
    Dönüş:
      {"A": 3xHxW [-1,1], "B": 1xHxW [-1,1], "path": (pathA, pathB)}
    """
    def __init__(
        self,
        rgb_root: str,
        thermal_root: str,
        image_size: int = 256,
        augment: bool = True,
        seed: int = 0,
    ):
        self.rgb_root = Path(rgb_root)
        self.thermal_root = Path(thermal_root)
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)

        rgb_files = [p for p in self.rgb_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        ther_files = [p for p in self.thermal_root.rglob("*") if p.suffix.lower() in IMG_EXTS]

        rgb_map = {p.stem: p for p in rgb_files}
        ther_map = {p.stem: p for p in ther_files}

        common = sorted(set(rgb_map.keys()) & set(ther_map.keys()))
        if len(common) == 0:
            raise RuntimeError(
                f"Eşleşen isim bulunamadı: {self.rgb_root} ve {self.thermal_root}"
            )

        self.pairs: List[Tuple[Path, Path]] = [(rgb_map[s], ther_map[s]) for s in common]

    def __len__(self) -> int:
        return len(self.pairs)

    def _paired_transforms(self, imgA: Image.Image, imgB: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Basit: her ikisini de kareye yeniden boyutlandır, isteğe bağlı yatay çevir.
        target = self.image_size
        imgA = TF.resize(imgA, [target, target], interpolation=IM.BICUBIC)
        imgB = TF.resize(imgB, [target, target], interpolation=IM.BICUBIC)

        if self.augment and self.rng.random() < 0.5:
            imgA = TF.hflip(imgA)
            imgB = TF.hflip(imgB)

        return imgA, imgB

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pA, pB = self.pairs[idx]
        imgA = Image.open(pA).convert("RGB")  # 3 kanal
        imgB = Image.open(pB)
        if imgB.mode not in ("L", "I;16", "I;16B", "I;16L"):
            imgB = imgB.convert("L")  # 1 kanal

        imgA, imgB = self._paired_transforms(imgA, imgB)

        tA = _to_tensor_01(imgA)  # 3xHxW
        tB = _to_tensor_01(imgB)  # 1xHxW (veya 16-bit normalize edilerek)

        if tA.shape[0] != 3:
            raise RuntimeError(f"A RGB 3 kanal bekleniyordu, gelen: {tA.shape}")
        if tB.ndim == 2:
            tB = tB.unsqueeze(0)
        if tB.shape[0] != 1:
            tB = tB[:1, ...]

        tA = _normalize_tanh(tA)
        tB = _normalize_tanh(tB)

        return {"A": tA, "B": tB, "path": (str(pA), str(pB))}


def make_paired_loader(
    rgb_root: str,
    thermal_root: str,
    image_size: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True,
) -> DataLoader:
    ds = PairedFoldersDataset(rgb_root, thermal_root, image_size=image_size, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
