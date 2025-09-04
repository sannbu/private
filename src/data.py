# data/dataloader.py
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode as IM


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# =========================
#  Ortak ABC dataset iskeleti
# =========================
class Pix2PixDataset(Dataset):
    def __init__(self, root, invert=False, transform=None, device=None):
        super().__init__()
        self.root = root
        self.invert = invert  # Orijinal anlam: A ve B'yi yer değiştir
        self.transform = transform  # (maps/facades/cityscapes/places) için kullanılıyor
        self.device = device
        self.filenames = sorted(os.listdir(root))

    def __len__(self):
        return len(self.filenames)

    def get_image_pair(self, filepath):
        raise NotImplementedError("Pix2PixDataset is an ABC. Use a concrete Dataset.")

    def __getitem__(self, index):
        filepath = os.path.join(self.root, self.filenames[index])
        image_a, image_b = self.get_image_pair(filepath)

        # invert == True -> A ve B yönünü değiştir
        if self.invert:
            image_b, image_a = image_a, image_b

        if self.transform is not None:
            # Aynı augmentasyonun A ve B'ye uygulanması için ortak seed
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            image_a = self.transform(image_a)
            random.seed(seed)
            image_b = self.transform(image_b)

        if self.device is not None:
            image_a = image_a.to(self.device)
            image_b = image_b.to(self.device)

        return image_a, image_b


# =========================
#  Orijinal tek-dosya veri kümeleri
# =========================
class MapsDataset(Pix2PixDataset):
    def get_image_pair(self, filepath):
        image = Image.open(filepath).convert('RGB')
        w, h = image.size
        image_a = image.crop((w // 2, 0, w, h))
        image_b = image.crop((0, 0, w // 2, h))
        return image_a, image_b

class PlacesDataset(Pix2PixDataset):
    def get_image_pair(self, filepath):
        image_b = Image.open(filepath).convert('RGB')
        image_a = image_b.convert('L')  # 1 kanal giriş
        return image_a, image_b

class CityscapesDataset(Pix2PixDataset):
    def get_image_pair(self, filepath):
        image = Image.open(filepath).convert('RGB')
        w, h = image.size
        image_b = image.crop((0, 0, w // 2, h))
        image_a = image.crop((w // 2, 0, w, h))
        return image_a, image_b

class FacadesDataset(Pix2PixDataset):
    def get_image_pair(self, filepath):
        image = Image.open(filepath).convert('RGB')
        w, h = image.size
        image_b = image.crop((0, 0, w // 2, h))
        image_a = image.crop((w // 2, 0, w, h))
        return image_a, image_b


# =========================
#  Yeni: A/B klasörlü RGB->Thermal dataset
#   root/
#     A/  (RGB)
#     B/  (Thermal, 8/16-bit)
# =========================
class ABFoldersDataset(Dataset):
    def __init__(self, root: str, train: bool = True, invert: bool = False,
                 resize: int = 286, crop: int = 256, device=None):
        super().__init__()
        self.root = Path(root)
        self.train = train
        self.invert = invert  # A ve B'yi yer değiştirir (RGB <-> Thermal) — dikkat!
        self.resize = resize
        self.crop = crop
        self.device = device

        A_root = self.root / "A"
        B_root = self.root / "B"
        a_files = [p for p in A_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        b_files = [p for p in B_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        a_map = {p.stem: p for p in a_files}
        b_map = {p.stem: p for p in b_files}
        common = sorted(set(a_map) & set(b_map))
        if not common:
            raise RuntimeError(f"Eşleşen dosya bulunamadı: {A_root} ve {B_root}")

        self.pairs: List[Tuple[Path, Path]] = [(a_map[s], b_map[s]) for s in common]

    def __len__(self):
        return len(self.pairs)

    # ---- yardımcılar ----
    @staticmethod
    def _to_tensor_rgb(img: Image.Image) -> torch.Tensor:
        # PIL RGB -> (3,H,W) float [0,1]
        return TF.to_tensor(img)

    @staticmethod
    def _to_tensor_gray(img: Image.Image) -> torch.Tensor:
        """
        PIL Gray (8/16-bit) -> (1,H,W) float [0,1]
        16-bit ise 65535'e bölüp 8-bit'e çevirmeden direkt tensöre alıyoruz.
        """
        if img.mode in ("I;16", "I;16B", "I;16L"):
            arr = np.array(img, dtype=np.uint16).astype(np.float32) / 65535.0
            if arr.ndim == 2:
                arr = arr[None, ...]  # 1xHxW
            else:
                arr = arr.transpose(2, 0, 1)  # olması beklenmez ama emniyet
            return torch.from_numpy(arr)
        else:
            t = TF.to_tensor(img)  # 1xHxW
            return t

    @staticmethod
    def _to_tanh(x: torch.Tensor) -> torch.Tensor:
        # [0,1] -> [-1,1]
        return x * 2.0 - 1.0

    def _resize_crop_flip(self, A: Image.Image, B: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.resize and self.resize > 0:
            A = TF.resize(A, [self.resize, self.resize], interpolation=IM.BICUBIC)
            B = TF.resize(B, [self.resize, self.resize], interpolation=IM.BICUBIC)

        if self.crop and self.crop > 0:
            if self.train:
                i, j, h, w = transforms.RandomCrop.get_params(A, (self.crop, self.crop))
                A = TF.crop(A, i, j, h, w)
                B = TF.crop(B, i, j, h, w)
            else:
                A = TF.center_crop(A, [self.crop, self.crop])
                B = TF.center_crop(B, [self.crop, self.crop])

        if self.train and random.random() < 0.5:
            A = TF.hflip(A)
            B = TF.hflip(B)

        return A, B

    def __getitem__(self, idx: int):
        pA, pB = self.pairs[idx]
        A = Image.open(pA).convert("RGB")
        B = Image.open(pB)
        if B.mode not in ("L", "I;16", "I;16B", "I;16L"):
            B = B.convert("L")

        # Senkron augment
        A, B = self._resize_crop_flip(A, B)

        tA = self._to_tensor_rgb(A)  # (3,H,W) [0,1]
        tB = self._to_tensor_gray(B) # (1,H,W) [0,1]

        # invert -> yön değişimi (dikkat: modelin in/out kanallarıyla uyumlu olmalı)
        if self.invert:
            tA, tB = tB, tA

        tA = self._to_tanh(tA)
        tB = self._to_tanh(tB)

        if self.device is not None:
            tA = tA.to(self.device)
            tB = tB.to(self.device)

        return tA, tB


# =========================
#  Dataloader fabrikası
# =========================
def dataloader(root, dataset, invert=False, device=None, train=True, resize=None, crop=None,
               batch_size=1, shuffle=False, num_workers=4):
    """
    root:
      - rgb2thermal için:  <data_split> klasörü (içinde A/ ve B/ olmalı)
      - diğerleri için:   doğrudan görüntü dosyalarının olduğu klasör
    """

    ds_name = (dataset or "").lower()

    if ds_name in ("rgb2thermal", "rgb-thermal", "rgbt"):
        ds = ABFoldersDataset(root=root, train=train, invert=invert,
                              resize=resize, crop=crop, device=device)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True, drop_last=True)

    # Aşağıdakiler orijinal tek-dosya veri kümeleri (transform ile)
    if ds_name == 'places':
        mean, std = (0.5,), (0.5,)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    if train:
        transform = transforms.Compose([
            transforms.Resize(resize, interpolation=IM.BICUBIC) if resize else transforms.Lambda(lambda x: x),
            transforms.RandomCrop(crop) if crop else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(crop, interpolation=IM.BICUBIC) if crop else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if ds_name == 'places':
        ds = PlacesDataset(root, invert, transform=transform, device=device)
    elif ds_name == 'maps':
        ds = MapsDataset(root, invert, transform=transform, device=device)
    elif ds_name == 'cityscapes':
        ds = CityscapesDataset(root, invert, transform=transform, device=device)
    elif ds_name == 'facades':
        ds = FacadesDataset(root, invert, transform=transform, device=device)
    else:
        raise AssertionError("Dataset not found. Use one among "
                             "'rgb2thermal | cityscapes | facades | maps | places'")

    return DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=True)
