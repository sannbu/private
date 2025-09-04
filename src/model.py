# src/pix2pix_rgb2thermal.py
import torch
from torch import nn
import modules


class Pix2Pix:
    """
    RGB -> Thermal (3 -> 1) Pix2Pix eğitim sarmalayıcısı.

    Notlar:
    - G/ D, modules.UNet ve modules.PatchGAN (logits döndüren) sürümü ile uyumludur.
    - Girişler [-1, 1] aralığına normalize edilmiş olmalı.
    - D.forward(cond, x) sırasını kullanıyoruz: cond=A(RGB), x=B(thermal).
      Eğer senin PatchGAN'in imzası (x, cond) ise çağrılarda argümanları yer değiştir.
    """
    def __init__(self, lr=2e-4, lambda_l1=100.0, lambda_d=1.0,
                 in_channels=3, out_channels=1, device=None):
        # Ağlar
        self.net_g = modules.UNet(in_channels=in_channels, out_channels=out_channels)
        self.net_d = modules.PatchGAN(in_channels=in_channels, out_channels=out_channels)

        # Ağırlık başlatma
        self.net_g.apply(self._init_weights)
        self.net_d.apply(self._init_weights)

        # Optimizasyon
        self.lr = lr
        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=lr, betas=(0.5, 0.999))

        # LR scheduler (şimdilik sabit)
        self.lr_lambda = lambda epoch: 1.0
        self.scheduler_g = torch.optim.lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda=self.lr_lambda)
        self.scheduler_d = torch.optim.lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda=self.lr_lambda)

        # Kayıplar
        self.adv_loss = nn.BCEWithLogitsLoss()   # D logits -> BCEWithLogits
        self.l1_loss = nn.L1Loss()

        # Ağırlıklar
        self.lambda_l1 = lambda_l1
        self.lambda_d = lambda_d

        # Cihaz
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # -------- utils --------
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if getattr(m, "bias", None) is not None and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def _gan_loss(self, logits, real: bool):
        target = torch.ones_like(logits) if real else torch.zeros_like(logits)
        return self.adv_loss(logits, target)

    def scheduler_step(self):
        self.scheduler_g.step()
        self.scheduler_d.step()

    def _apply(self, fn):
        # Ağları taşı
        fn(self.net_g)
        fn(self.net_d)
        # Optimizer'ları yeniden oluştur (cihaz değişimlerinde pratik)
        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=self.lr, betas=(0.5, 0.999))
        # State içindeki tensörleri de taşı
        for optimizer in (self.optimizer_g, self.optimizer_d):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = fn(value)

    def cpu(self):
        self._apply(lambda t: t.cpu())
        self.device = "cpu"
        return self

    def to(self, device=None):
        if device is None:
            device = self.device
        self._apply(lambda t: t.to(device))
        self.device = device
        return self

    # -------- train / eval step --------
    def train(self, batch):
        """
        batch: (x, y)
          x: A (RGB, N x 3 x H x W)  [-1,1]
          y: B (Thermal, N x 1 x H x W) [-1,1]
        """
        self.net_g.train()
        self.net_d.train()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # --------- D ---------
        self.optimizer_d.zero_grad()
        with torch.no_grad():
            fake = self.net_g(x)

        # D(cond=A, x=B)
        real_logits = self.net_d(x, y)
        fake_logits = self.net_d(x, fake.detach())

        loss_d_real = self._gan_loss(real_logits, True) * self.lambda_d
        loss_d_fake = self._gan_loss(fake_logits, False) * self.lambda_d
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.optimizer_d.step()

        # --------- G ---------
        self.optimizer_g.zero_grad()
        fake = self.net_g(x)
        fake_logits = self.net_d(x, fake)

        loss_g_gan = self._gan_loss(fake_logits, True)
        loss_g_l1 = self.l1_loss(fake, y) * self.lambda_l1
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        self.optimizer_g.step()

        return {
            'g': loss_g.item(),
            'g_gan': loss_g_gan.item(),
            'g_l1': loss_g_l1.item(),
            'd': loss_d.item(),
            'd_real': loss_d_real.item(),
            'd_fake': loss_d_fake.item(),
        }, fake.detach()

    @torch.no_grad()
    def eval(self, batch):
        self.net_g.eval()
        self.net_d.eval()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        fake = self.net_g(x)
        real_logits = self.net_d(x, y)
        fake_logits = self.net_d(x, fake)

        loss_d_real = self._gan_loss(real_logits, True) * self.lambda_d
        loss_d_fake = self._gan_loss(fake_logits, False) * self.lambda_d
        loss_d = loss_d_real + loss_d_fake

        loss_g_gan = self._gan_loss(fake_logits, True)
        loss_g_l1 = self.l1_loss(fake, y) * self.lambda_l1
        loss_g = loss_g_gan + loss_g_l1

        return {
            'g': loss_g.item(),
            'g_gan': loss_g_gan.item(),
            'g_l1': loss_g_l1.item(),
            'd': loss_d.item(),
            'd_real': loss_d_real.item(),
            'd_fake': loss_d_fake.item(),
        }, fake

    # -------- checkpoint helpers --------
    def named_components(self):
        yield 'generator', self.net_g
        yield 'discriminator', self.net_d
        yield 'generator_optimizer', self.optimizer_g
        yield 'discriminator_optimizer', self.optimizer_d
        yield 'generator_scheduler', self.scheduler_g
        yield 'discriminator_scheduler', self.scheduler_d

    def state_dict(self):
        return {name: comp.state_dict() for name, comp in self.named_components()}

    def load_state_dict(self, state_dict):
        for name, comp in self.named_components():
            comp.load_state_dict(state_dict[name])
