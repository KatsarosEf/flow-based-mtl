import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

class ECAModule(nn.Module):
    def __init__(self, channel):
        super(ECAModule, self).__init__()
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=3, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, norm=True, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, norm=False, relu=False))
    def forward(self, x):
        return self.main(x) + x

class FFTResBLock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FFTResBLock, self).__init__()
        self.local_stream = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.global_stream = nn.Sequential(
         nn.Conv2d(in_channel*2, out_channel*2, kernel_size=1, padding=0, bias=False),
         nn.ReLU(),
         nn.Conv2d(in_channel*2, out_channel*2, kernel_size=1, padding=0, bias=False),
      )

    def forward(self, x):
        lcl_feats = self.local_stream(x)
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm='backward')
        y = torch.cat([y.imag, y.real], dim=1)
        y = self.global_stream(y)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        glb_feats = torch.fft.irfft2(y, s=(H, W), norm='backward')

        return lcl_feats + glb_feats + x

class EBlock(nn.Module):
    def __init__(self, channel, num_res=5, name='fft'):
        super(EBlock, self).__init__()
        if name == 'res':
            layers = [ResBlock(channel, channel) for _ in range(num_res)]
        elif name == 'fft':
            layers = [FFTResBLock(channel, channel) for _ in range(num_res)]
        elif name == 'inverted':
            layers = [SparseLocalBlock(channel) for _ in range(num_res)]
        elif name == 'inverted_fft':
            layers = [Block(channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=2, name='fft'):
        super(DBlock, self).__init__()
        if name == 'res':
            layers = [ResBlock(channel, channel) for _ in range(num_res)]
        elif name == 'fft':
            layers = [FFTResBLock(channel, channel) for _ in range(num_res)]
        elif name == 'inverted':
            layers = [SparseLocalBlock(channel) for _ in range(num_res)]
        elif name == 'inverted_fft':
            layers = [Block(channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class FAM_homo(nn.Module):
    def __init__(self, channel):
        super(FAM_homo, self).__init__()
        self.conv = BasicConv(channel, channel//2, kernel_size=1, stride=1, relu=False)
        self.fe = BasicConv(5, channel // 2, kernel_size=3, stride=1, relu=False)

    def forward(self, feats, mask, restred_img):
        mask = torch.nn.functional.softmax(mask, dim=1)
        att_feats = self.conv(feats)
        img_feats = self.fe(torch.cat([restred_img, mask], 1))
        out = torch.cat([img_feats, att_feats], 1)
        return out

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out



class SD(nn.Module):
    def __init__(self, channel):
        super(SD, self).__init__()
        self.conv1 = BasicConv(channel, channel, kernel_size=3, stride=1, norm=True, relu=True)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = BasicConv(channel, channel//2, kernel_size=3, stride=1, norm=True, relu=True)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.convout = BasicConv(channel//2, 2, kernel_size=3, stride=1, norm=False, relu=False)
    def forward(self, x):
        x = self.pool1(self.conv1(x) + x)
        x = self.pool2(self.conv2(x))
        x = F.interpolate(self.convout(x), scale_factor=4.0)
        return x

class DD(nn.Module):
    def __init__(self, channel):
        super(DD, self).__init__()
        self.conv = BasicConv(channel, channel, kernel_size=3, stride=1, norm=True, relu=True)
        self.conv_out = BasicConv(channel, 3, kernel_size=3, stride=1, norm=False, relu=False)

    def forward(self, x):
        x = self.conv(x) + x
        x = self.conv_out(x)
        return x

class SparseLocalBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim,  layer_scale_init_value=1e-6):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        #self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4*dim, kernel_size=1, padding=0)
        self.act = nn.ReLU()
        self.pwconv2 = nn.Conv2d(4*dim, dim, kernel_size=1, padding=0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        x = input + x
        return x

class FFTBLock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(FFTBLock, self).__init__()
        self.global_stream = nn.Sequential(
         nn.Conv2d(in_channel*2, out_channel*2, kernel_size=1, padding=0, bias=False),
         nn.ReLU(),
         nn.Conv2d(in_channel*2, out_channel*2, kernel_size=1, padding=0, bias=False))

    def forward(self, x):
        ffted = torch.rfft(x, signal_ndim=2)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((x.shape[0], -1,) + ffted.size()[3:])
        ffted = self.global_stream(ffted)
        ffted = ffted.view((x.shape[0], -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        glb_feats = torch.irfft(ffted, signal_ndim=2, normalized=False, signal_sizes=x.shape[-2:])

        return glb_feats

class Block(nn.Module):

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.local_block = SparseLocalBlock(dim, layer_scale_init_value)
        self.global_block = FFTBLock(dim, dim)

    def forward(self, x):
        local_feats = self.local_block(x)
        global_feats = self.global_block(x)
        return local_feats + global_feats


class GSA(nn.Module):
    def __init__(self, feats):
        super(GSA, self).__init__()
        self.n_feats = feats

        self.F_f = nn.Sequential(
            nn.Conv2d(2 * self.n_feats, 2 * self.n_feats, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.n_feats, self.n_feats, kernel_size=1),
            nn.Sigmoid())
        # out channel: 128
        self.F_p = nn.Sequential(
            nn.Conv2d(2 * self.n_feats, 2 * self.n_feats, kernel_size=1),
            nn.Conv2d(2 * self.n_feats, self.n_feats, kernel_size=1)
        )
        # fusion layer
        self.fusion = nn.Conv2d(2*self.n_feats, self.n_feats, kernel_size=1)

    def forward(self, f_curr, f_prv):
        cor = torch.cat([f_curr, f_prv], 1)
        gap = F.adaptive_avg_pool2d(cor, (1, 1))
        f_f = self.F_f(gap) #out 128
        f_p = self.F_p(cor) #out 128

        f_prv = f_f*f_p
        out = self.fusion(torch.cat([f_curr, f_prv], 1))
        return out