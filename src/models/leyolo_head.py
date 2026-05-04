import math
import torch
import torch.nn as nn
from ultralytics.utils.tal import make_anchors

from models.mobilevit_xxs import MobileViT, model_cfg_xxs

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


"""
Backbone must provide 3 semantic feature levels: P3, P4, P5.
This MobileViT-XXS backbone returns:
  P3: stage2 output (stride 8, 48 channels)
  P4: stage3 output (stride 16, 64 channels)
  P5: stage4 output (stride 32, 320 channels)
"""
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class MobileViTXXSBackbone(nn.Module):
    def __init__(self, image_size: int = 256):
        super().__init__()
        cfg = model_cfg_xxs
        self.backbone = MobileViT(
            image_size,
            cfg["features"],
            cfg["d"],
            cfg["layers"],
            cfg["expansion_ratio"],
            num_classes=1000,
        )

    def forward(self, x):
        x = self.backbone.stem(x)
        x = self.backbone.stage1(x)
        p3 = self.backbone.stage2(x)
        p4 = self.backbone.stage3(p3)
        p5 = self.backbone.stage4(p4)
        return [p3, p4, p5]


class mn_conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        padding = 0 if k == s else autopad(k, p, d)
        self.c = nn.Conv2d(c1, c2, k, s, padding, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU()
        # self.act = activation_function(act)

    def forward(self, x):
        return self.act(self.bn(self.c(x)))


class LeNeckBlock(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, stride=1, pw=True):
        super().__init__()

        c_mid = e if e is not None else c1
        self.residual = c1 == c2 and stride == 1

        layers = []
        if pw and c_mid != c1:
            layers.extend([
                nn.Conv2d(c1, c_mid, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_mid),
                nn.SiLU(),
            ])

        layers.extend([
            nn.Conv2d(c_mid, c_mid, kernel_size=k, stride=stride, padding=k // 2, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),
            nn.Conv2d(c_mid, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = self.layers(x)
        if self.residual:
            return out + x
        return out


class LeNeck(torch.nn.Module):
    def __init__(self, width, multiplier=1.5, depth=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)

        self.p4_up = [LeNeckBlock(c1=width[1] + width[2], c2=int(64 * multiplier), e=int(128 * multiplier), k=5)]
        self.p4_up += [LeNeckBlock(c1=int(64 * multiplier), c2=int(64 * multiplier), e=int(128 * multiplier), k=5)
                       for _ in range(int(2 * depth))]
        self.p4_up = nn.Sequential(*self.p4_up)

        self.p3_up = [LeNeckBlock(c1=int(64 * multiplier) + width[0], c2=int(32 * multiplier), e=int(64 * multiplier) + width[0], k=3, pw=False)]
        self.p3_up += [LeNeckBlock(c1=int(32 * multiplier), c2=int(32 * multiplier), e=int(96 * multiplier), k=3)
                       for _ in range(int(2 * depth))]
        self.p3_up = nn.Sequential(*self.p3_up)

        self.p3_downsampling = mn_conv(int(32 * multiplier), int(64 * multiplier), k=3, s=2, p=1)

        self.p4_down = [LeNeckBlock(c1=int(64 * multiplier) + int(64 * multiplier), c2=int(64 * multiplier), e=int(128 * multiplier), k=5)]
        self.p4_down += [LeNeckBlock(c1=int(64 * multiplier), c2=int(64 * multiplier), e=int(128 * multiplier), k=5)
                         for _ in range(int(2 * depth))]
        self.p4_down = nn.Sequential(*self.p4_down)
        self.p4_downsampling = mn_conv(int(64 * multiplier), int(96 * multiplier), k=3, s=2, p=1)

        self.p5_down = [LeNeckBlock(c1=int(96 * multiplier) + width[2], c2=int(96 * multiplier), e=int(96 * multiplier) + width[2], k=5)]
        self.p5_down += [LeNeckBlock(c1=int(96 * multiplier), c2=int(96 * multiplier), e=int(192 * multiplier), k=5)
                         for _ in range(int(2 * depth))]
        self.p5_down = nn.Sequential(*self.p5_down)

    def forward(self, x):
        p3, p4, p5 = x

        p5_up = torch.cat(tensors=[self.up(p5), p4], dim=1) #P5 ----> to P4 (no computation at P5 in the down-up passage)
        p4_up = self.p4_up(p5_up)    #P4_UP concat with P4_Backbone ----> #P4_UP conv computations  (save for later, P4 top-down passage)

        p3_up = torch.cat(tensors=[self.up(p4_up), p3], dim=1)  #P4_UP ---> P3 and concat with P3 backbone
        p3_up = self.p3_up(p3_up)                   #P3 computation (save for later, no computation at P3 top-down passage, NECK OUTPUT)
        p4_down = self.p3_downsampling(p3_up)       #P3_UP ---> P4_DOWN (save for NECK OUTPUT)

        p4_down = torch.cat(tensors=[p4_down, p4_up], dim=1) #Concat with P4_UP from the down-up passage
        p4_down = self.p4_down(p4_down) #P4_DOWN Computation (save for later, NECK OUTPUT)

        p5_down = self.p4_downsampling(p4_down) #p4_DOWN --> P5_DOWN (and last neck computation)
        p5_down = torch.cat(tensors=[p5_down, p5], dim=1)
        p5_down = self.p5_down(p5_down)         #P5_DOWN Computation (save for NECK OUTPUT)

        return p3_up, p4_down, p5_down


class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class LeHead(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16    # DFL channels
        self.nc = nc    # number of classes
        self.nl = len(filters)        # number of detection layers
        self.no = nc + self.ch * 4    # number of outputs per anchor
        self.stride = torch.zeros(self.nl)    # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box, k=1),
                                                           Conv(box, box, k=3, p=1, g=box),
                                                           Conv(box, box, k=3, p=1, g=box),
                                                           torch.nn.Conv2d(box, out_channels=4 * self.ch,
                                                                           kernel_size=1)) for x in filters)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, cls, k=1),
                                                           Conv(cls, cls, k=3, p=1, g=cls),
                                                           Conv(cls, cls, k=3, p=1, g=cls),
                                                           torch.nn.Conv2d(cls, out_channels=self.nc,
                                                                           kernel_size=1)) for x in filters)

    def forward(self, x):
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            #  cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class LeYOLO(torch.nn.Module):
    def __init__(self, num_classes, multiplier=1.5, depth=2, image_size: int = 256):
        super().__init__()
        self.net = MobileViTXXSBackbone(image_size=image_size)
        width = [48, 64, 320]   # P3, P4, P5 channels for MobileViT-XXS
        self.fpn = LeNeck(width, multiplier=multiplier, depth=depth)    # Special width for MobileViT-XXS backbone

        img_dummy = torch.zeros(1, 3, image_size, image_size)
        self.head = LeHead(num_classes, (int(32 * multiplier), int(64 * multiplier), int(96 * multiplier)))
        self.head.stride = torch.tensor([image_size / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def leyolo_n(num_classes: int = 80, multiplier: float = 1.5, depth: int = 2, image_size: int = 256):
    return LeYOLO(num_classes, multiplier=multiplier, depth=depth, image_size=image_size)


if __name__ == "__main__":
    model = leyolo_n(80)
    print(model)
    print("params:", sum(p.numel() for p in model.parameters()))
