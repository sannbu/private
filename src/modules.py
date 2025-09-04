import torch
import torch.nn as nn

class EncoderDecoderBlock(nn.Module):
    def __init__(self, module_type, in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1,
                 bias=False, dropout_p=0.0, norm=True, activation=True):
        super().__init__()

        self.module_type = module_type
        # Pix2Pix’te: encoder -> LeakyReLU; decoder -> ReLU (daha klasik)
        if module_type == 'encoder':
            self.act = nn.LeakyReLU(0.2, inplace=True) if activation else None
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, bias=bias)
        elif module_type == 'decoder':
            self.act = nn.ReLU(inplace=True) if activation else None
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, bias=bias)
        else:
            raise NotImplementedError(f"Module type '{module_type}' is not valid")

        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else None

    def forward(self, x):
        if self.act is not None:
            x = self.act(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UNet(nn.Module):
    """
    RGB (3ch) -> Thermal (1ch) U-Net
    """
    def __init__(self, in_channels=3, out_channels=1, bias=False, dropout_p=0.5, norm=True):
        super().__init__()
        # 7 down + 7 up (256 -> 1 ölçek)
        self.encoders = nn.ModuleList([
            EncoderDecoderBlock('encoder', in_channels, 64,  bias=bias, norm=False, activation=False),
            EncoderDecoderBlock('encoder', 64,  128, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 128, 256, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 256, 512, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 512, 512, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 512, 512, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 512, 512, bias=bias, norm=False)  # bottleneck
        ])

        # İlk 3 up-block’a dropout ver (pix2pix stilinde)
        self.decoders = nn.ModuleList([
            EncoderDecoderBlock('decoder', 512,   512, bias=bias, norm=norm, dropout_p=dropout_p),
            EncoderDecoderBlock('decoder', 1024,  512, bias=bias, norm=norm, dropout_p=dropout_p),
            EncoderDecoderBlock('decoder', 1024,  512, bias=bias, norm=norm, dropout_p=dropout_p),
            EncoderDecoderBlock('decoder', 1024,  256, bias=bias, norm=norm),
            EncoderDecoderBlock('decoder', 512,   128, bias=bias, norm=norm),
            EncoderDecoderBlock('decoder', 256,    64, bias=bias, norm=norm),
            EncoderDecoderBlock('decoder', 128, out_channels, bias=bias, norm=False)  # son: BN yok
        ])

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Down path
        feats = [x]
        for enc in self.encoders:
            feats.append(enc(feats[-1]))  # len = 1 + 7 = 8

        # Up path (skip’lerle)
        output = None
        # reversed(feats) = [bottleneck, e6, e5, e4, e3, e2, e1, input]
        for i, (dec, skip) in enumerate(zip(self.decoders, reversed(feats))):
            if i == 0:
                # bottleneck’ten başla (skip = bottleneck)
                output = dec(skip)
            else:
                output = dec(torch.cat([output, skip], dim=1))

        return self.tanh(output)  # [-1, 1]


class PatchGAN(nn.Module):
    """
    Koşullu ayrımcı: input = cat([cond(A), target_or_fake(B)], dim=1)
    Giriş kanal sayısı: in_channels + out_channels (RGB+Thermal=3+1=4)
    Çıkış: logits (Sigmoid YOK) -> BCEWithLogitsLoss ile kullanılacak.
    """
    def __init__(self, in_channels=3, out_channels=1, bias=False, norm=True):
        super().__init__()
        cin = in_channels + out_channels  # 3 + 1 = 4

        self.blocks = nn.ModuleList([
            EncoderDecoderBlock('encoder', cin,   64, bias=bias, norm=False, activation=False),
            EncoderDecoderBlock('encoder', 64,   128, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 128,  256, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 256,  512, bias=bias, norm=norm, stride=1),
            EncoderDecoderBlock('encoder', 512,    1, bias=bias, norm=False, stride=1)
        ])

    def forward(self, cond, x):
        out = torch.cat([cond, x], dim=1)
        for blk in self.blocks:
            out = blk(out)
        return out  # logits
