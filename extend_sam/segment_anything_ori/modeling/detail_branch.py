import torch
from torch import nn


class SharedSpatialAttention(nn.Module):
    """Position linear attention"""

    def __init__(self, in_places, out_channel, eps=1e-6):
        super().__init__()  # 初始化父类
        self.in_places = in_places  # 输入 channel
        self.l2_norm = l2_norm  # L2范数
        self.eps = eps  # 防 nan 参数
        # QKV生成卷积
        self.out = out_channel
        self.query_conv = nn.Conv2d(in_places, out_channel // 4, 1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=out_channel // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=out_channel, kernel_size=1)
        self.l1 = nn.Sequential(nn.Conv2d(in_places, out_channel, 3, 1, 1),
                                nn.BatchNorm2d(out_channel, momentum=0.1), nn.ReLU(inplace=True))
        self.l2 = nn.Sequential(nn.Conv2d(in_places, out_channel, kernel_size=1),
                                nn.BatchNorm2d(out_channel, momentum=0.1), nn.ReLU(inplace=True))
        self.out_cov = ConvBNRelu(out_channel, out_channel, 1, 1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, _, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)
        Q = self.l2_norm(Q).permute(-3, -1, -2)  # 对Q进行L2正则
        K = self.l2_norm(K)  # 对K进行L2正则
        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))  # 下方全部
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, self.out, width * height)
        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)  # 上方全部
        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, self.out, height, width) + self.l1(x) + self.l2(x)
        return self.out_cov(weight_value)


class SharedChannelAttention(nn.Module):
    """Channel linear attention"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.l2_norm = l2_norm
        self.eps = eps

    def forward(self, x):
        batch_size, channels, width, height = x.shape
        Q = x.view(batch_size, channels, -1)
        K = x.view(batch_size, channels, -1)
        V = x.view(batch_size, channels, -1)

        Q = self.l2_norm(Q)
        K = self.l2_norm(K).permute(-3, -1, -2)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bn->bc", K, torch.sum(Q, dim=-2) + self.eps))

        value_sum = torch.einsum("bcn->bn", V).unsqueeze(-1).permute(0, 2, 1)
        value_sum = value_sum.expand(-1, channels, width * height)
        matrix = torch.einsum('bcn, bnm->bcm', V, K)
        matrix_sum = value_sum + torch.einsum("bcm, bmn->bcn", matrix, Q)
        weight_value = torch.einsum("bcn, bc->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, channels, height, width)

        return weight_value


class AFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.ca = SharedChannelAttention()
        self.sa = SharedSpatialAttention(in_channel, in_channel)
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                                  nn.BatchNorm2d(in_channel),
                                  nn.ReLU(inplace=True))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        w = self.gamma * x + (1 - self.gamma) * y
        return self.conv(self.sa(w) + self.ca(w))


class TwoWayConv(nn.Module):
    def __int__(self, inpt, outpt):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inpt, outpt, 3, 1, 1),
                                   nn.BatchNorm2d(outpt),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(inpt, outpt, 1, 1),
                                   nn.BatchNorm2d(outpt),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv1(x) + self.conv2(x)


class DarkChannelPrompter(nn.Module):
    def __int__(self, dims):
        super().__init__()
        self.stem = nn.Sequential(TwoWayConv(1, 16),
                                  TwoWayConv(16, 64),
                                  TwoWayConv(64, dims))
