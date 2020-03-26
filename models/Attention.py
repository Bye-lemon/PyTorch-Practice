import torch
import torch.nn as nn
import torch.nn.functional as F

WATCH = lambda x: print(x.shape)


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels // ratio, out_channels=channels, kernel_size=1)
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.mlp(avg_out)
        max_out = self.max_pool(x)
        max_out = self.mlp(max_out)
        return F.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, **kwargs)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return F.sigmoid(out)


class CBAMConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=16, **kwargs):
        super(CBAMConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.ca = ChannelAttention(channels=out_channels, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=7, padding=3)

    def forward(self, x):
        x = self.conv(x)
        x *= self.ca(x)
        x *= self.sa(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, channels, ratio=16):
        super(ChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels // ratio, out_channels=channels, kernel_size=1)
        )
        self.bn = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        return self.bn(self.mlp(self.avg_pool(x)))


class SpatialGate(nn.Module):
    def __init__(self, channels, ratio):
        super(SpatialGate, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // ratio, kernel_size=1),
            nn.Conv2d(in_channels=channels // ratio, out_channels=channels // ratio, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=channels // ratio, out_channels=channels // ratio, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=channels // ratio, out_channels=1, kernel_size=1)
        )
        self.bn = nn.BatchNorm2d(num_features=1)

    def forward(self, x):
        return self.bn(self.conv(x))


class BAMConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=16, **kwargs):
        super(BAMConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.cg = ChannelGate(channels=out_channels, ratio=ratio)
        self.sg = SpatialGate(channels=out_channels, ratio=ratio)

    def forward(self, x):
        x = self.conv(x)
        cg_out = self.cg(x)
        sg_out = self.sg(x)
        att = F.sigmoid(sg_out.repeat(1, cg_out.shape[1], 1, 1) + cg_out)
        x *= 1 + att
        return x


class AttentionAugmentedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dim_k, dim_v, num_h, **kwargs):
        super(AttentionAugmentedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_h = num_h
        self.dim_k_per_head = self.dim_k // self.num_h
        assert self.num_h != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dim_k % self.num_h == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dim_v % self.num_h == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        self.key_rel_w = None
        self.key_rel_h = None
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels - dim_v, **kwargs)
        self.conv_qkv = nn.Conv2d(in_channels=in_channels, out_channels=2 * dim_k + dim_v, **kwargs)
        self.conv_att = nn.Conv2d(in_channels=dim_v, out_channels=dim_v, kernel_size=1)

    def forward(self, x):
        mat_qkv = self.conv_qkv(x)
        batch, channel, height, width = mat_qkv.shape
        q, k, v = torch.split(mat_qkv, [self.dim_k, self.dim_k, self.dim_v], dim=1)
        q, k, v = tuple(map(
            lambda tensor: tensor.view(tensor.shape[0], self.num_h, tensor.shape[1] // self.num_h, tensor.shape[2],
                                       tensor.shape[3]),
            (q, k, v)
        ))
        q *= self.dim_k_per_head ** -0.5
        q_flat = q.view(batch, self.num_h, self.dim_k // self.num_h, height * width)
        k_flat = k.view(batch, self.num_h, self.dim_k // self.num_h, height * width)
        v_flat = v.view(batch, self.num_h, self.dim_v // self.num_h, height * width)
        logits = torch.matmul(q_flat.transpose(dim0=2, dim1=3), k_flat)
        logits += self.get_relative(q)
        weights = F.softmax(logits, dim=-1)
        att = torch.matmul(weights, v_flat.transpose(dim0=2, dim1=3))
        att = att.view(batch, self.dim_v, height, width)
        att = self.conv_att(att)
        x = self.conv(x)
        return torch.cat([x, att], dim=1)

    def get_relative(self, q):
        batch, num_h, dim_k, height, width = q.shape
        q = q.transpose(2, 4).transpose(2, 3)
        self.key_rel_w = nn.Parameter(torch.randn((2 * width - 1, self.dim_k_per_head)), requires_grad=True)
        self.key_rel_h = nn.Parameter(torch.randn((2 * height - 1, self.dim_k_per_head)), requires_grad=True)
        rel_logits_w = self._get_relative_1d(q, self.key_rel_w, height, width, num_h, "w")
        rel_logits_h = self._get_relative_1d(torch.transpose(q, 2, 3), self.key_rel_h, height, width, num_h, "h")
        return rel_logits_h + rel_logits_w

    def _get_relative_1d(self, q, key_rel, height, width, num_h, target):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, key_rel)
        rel_logits = rel_logits.view(-1, num_h * height, width, 2 * width - 1)
        batch = rel_logits.shape[0]
        padding = torch.zeros((batch, num_h * height, width, 1)).to(rel_logits)
        rel_logits = torch.cat([rel_logits, padding], dim=-1)
        rel_logits = rel_logits.view(batch, num_h * height, 2 * width ** 2)
        padding = torch.zeros((batch, num_h * height, width - 1)).to(rel_logits)
        rel_logits = torch.cat([rel_logits, padding], dim=-1)
        rel_logits = rel_logits.view(batch, num_h * height, width + 1, 2 * width - 1)
        rel_logits = rel_logits[:, :, : width, width - 1:]
        rel_logits = rel_logits.view(-1, num_h, height, width, width)
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, height, 1, 1))
        if target == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif target == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, num_h, height * width, height * width))
        return rel_logits
