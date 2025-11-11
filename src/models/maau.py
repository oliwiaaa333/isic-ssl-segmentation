# implementacja na podstawie publikacji PMC10137492 z malymi zmianami (np zamiast TensorFlow PyTorch)


from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# definicje konwolucji 3x3 (padding 1; zachowanie wymiarow), konwolucji 1x1 (mieszanie kanalow), funkcji inicjalizujacej wagi He

def conv3x3(in_ch, out_ch, stride=1, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

def init_weights_he(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)



# warstwa antyaliasingu (BlurPool), stosuje filtr Gaussa 3x3 w trybie depthwise convolution (kazdy kanal osobno), próbkowanie stride=2
# zastapienie klasycznego MaxPoolingu i unikniecie szarpania (mamy lekkie rozmycie obrazu, a potem probkowanie)

class BlurPool(nn.Module):
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        # 3x3 jadro gaussa (1/16) * [[1,2,1],[2,4,2],[1,2,1]]
        kernel = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]]) / 16.0
        kernel = kernel.view(1, 1, 3, 3)
        kernel = kernel.repeat(channels, 1, 1, 1)  # depthwise
        self.register_buffer("kernel", kernel) #zapisanie filtra jako stalego tensora
        self.groups = channels
        self.pad = 1

    #konwolucja 2D
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.pad, groups=self.groups)


# bramka uwagi; filtruje informacje przekazywane przez polaczenia skip miedzy encoderem a decoderem
# dla kazdego piksela obliczanie maski, ktora okresla na ile dana cecha jest istotna dla rekonstrukcji zmiany
# istotne obszary sa wzmacniane, a nieistotne tlumione

class AttentionGate(nn.Module):
    def __init__(self, in_ch_x: int, in_ch_g: int, inter_ch: Optional[int] = None):
        super().__init__()
        if inter_ch is None:
            inter_ch = max(in_ch_x // 2, 1)
        self.theta_x = conv1x1(in_ch_x, inter_ch, bias=False)
        self.phi_g   = conv1x1(in_ch_g, inter_ch, bias=True)
        self.psi     = conv1x1(inter_ch, 1, bias=True)
        self.act     = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # Dopasuj rozdzielczość (g może być mniejszy – upsample do x)
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)

        q = self.theta_x(x) + self.phi_g(g)
        q = self.act(q)
        alpha = self.sigmoid(self.psi(q))  # [B,1,H,W]
        return x * alpha


# bloki konwolucyjne
# umozliwia ekstrakcje cech lokalnych
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_ch, out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv3x3(out_ch, out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# blok kodujacy; stopniowo zmniejsza rozmiar obrazu
class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_antialias: bool = True):
        super().__init__()
        self.conv_block = DoubleConv(in_ch, out_ch)
        self.down = BlurPool(out_ch, stride=2) if use_antialias else nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv_block(x)   # cecha do skipa
        x = self.down(feat)     # zejście w dół skali
        return feat, x


# blok dekodujacy; stopniowo zwieksza rozmiar obrazu
class UpBlock(nn.Module):
    def __init__(self, in_ch_dec: int, in_ch_skip: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.att = AttentionGate(in_ch_skip, in_ch_dec) if use_attention else None
        self.conv_block = DoubleConv(in_ch_dec + in_ch_skip, out_ch)

    def forward(self, x_dec: torch.Tensor, x_skip: torch.Tensor):
        x_dec = self.up(x_dec)
        # dopasowanie rozdzielczosci (ochrona przed różnicami rounding)
        if x_dec.shape[-2:] != x_skip.shape[-2:]:
            x_dec = F.interpolate(x_dec, size=x_skip.shape[-2:], mode="bilinear", align_corners=False)

        if self.att is not None:
            x_skip = self.att(x_skip, x_dec)

        x = torch.cat([x_dec, x_skip], dim=1)
        return self.conv_block(x)


# model maau

class MAAU(nn.Module):
    """
    Multi-scale Anti-Aliasing Attention U-Net
    Parametry:
      in_channels: liczba kanałów wejścia (RGB=3)
      out_channels: liczba kanałów wyjścia (dla binarnej segmentacji: 1)
      encoder_filters: np. [64,128,256,512]
      decoder_filters: np. [512,256,128,64]
      use_attention_gates: czy stosować AG na skipach
      use_antialiasing_pool: czy stosować BlurPool w downsamplingu
      final_activation: None | "sigmoid"
    """
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            encoder_filters: List[int] = [64, 128, 256, 512],
            decoder_filters: List[int] = [512, 256, 128, 64],
            use_attention_gates: bool = True,
            use_antialiasing_pool: bool = True,
            final_activation: Optional[str] = "sigmoid",
    ):
        super().__init__()
        assert len(encoder_filters) == len(decoder_filters), \
            "Dla klasycznego 4-poziomowego U-Neta listy encoder/decoder powinny mieć tę samą długość."

        self.use_attention_gates = use_attention_gates
        self.final_activation = final_activation

        # encoder
        enc_ch = [in_channels] + encoder_filters
        self.down_blocks = nn.ModuleList([
            DownBlock(enc_ch[i], enc_ch[i+1], use_antialias=use_antialiasing_pool)
            for i in range(len(encoder_filters))
        ])

        # bottleneck (na najnizszym poziomie)
        self.bottleneck = DoubleConv(encoder_filters[-1], encoder_filters[-1] * 2)

        # decoder
        dec_in = [encoder_filters[-1] * 2] + decoder_filters[:-1]
        self.up_blocks = nn.ModuleList([
            UpBlock(in_ch_dec=dec_in[i], in_ch_skip=encoder_filters[-(i+1)], out_ch=decoder_filters[i],
                    use_attention=use_attention_gates)
            for i in range(len(decoder_filters))
        ])

        # wyjscie
        self.head = nn.Sequential(
            conv1x1(decoder_filters[-1], out_channels, bias=True)
        )

        # inicjalizacja wag
        self.apply(init_weights_he)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder (zbieramy skipy)
        skips = []
        h = x
        for down in self.down_blocks:
            feat, h = down(h)
            skips.append(feat)

        # bottleneck
        h = self.bottleneck(h)

        # decoder (odwrocone skipy)
        for i, up in enumerate(self.up_blocks):
            skip = skips[-(i+1)]
            h = up(h, skip)

        logits = self.head(h)

        if self.final_activation is None:
            return logits
        if self.final_activation.lower() == "sigmoid":
            return torch.sigmoid(logits)
        raise ValueError(f"Nieobsługiwana final_activation: {self.final_activation}")

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
