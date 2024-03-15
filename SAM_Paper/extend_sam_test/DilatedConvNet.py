import torch
import torch.nn as nn


class ThreeLayerDilatedConvNet(nn.Module):
    def __init__(self, channels):
        super(ThreeLayerDilatedConvNet, self).__init__()

        # 第一层卷积，空洞率为1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        # 第二层卷积，空洞率为2
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        # 第三层卷积，空洞率为5
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=5, dilation=5)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        return x+x3


class FourLayerDilatedConvNetA(nn.Module):
    def __init__(self, channels):
        super(FourLayerDilatedConvNetA, self).__init__()

        # 第一层卷积，空洞率为1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        # 第二层卷积，空洞率为2
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        # 第三层卷积，空洞率为5
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=5, dilation=5)
        # 第四层卷积，空洞率为8
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=8, dilation=8)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x+x4

# 多加了一个残差连接怕跑不起来
class FourLayerDilatedConvNetB(nn.Module):
    def __init__(self, channels):
        super(FourLayerDilatedConvNetB, self).__init__()

        # 第一层卷积，空洞率为1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        # 第二层卷积，空洞率为2
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        # 第三层卷积，空洞率为5
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=5, dilation=5)
        # 第四层卷积，空洞率为8
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=8, dilation=8)
        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.sigmoid(x2)
        x2 = x2+x

        x3 = self.conv3(x2)
        x3 = self.relu(x3)

        x4 = self.conv4(x3)
        x4 = self.sigmoid(x4)
        x4 = x4+x2

        return x4


# # 创建模型实例
# model = FourLayerDilatedConvNetB( channels =7)
#
# input_tensor = torch.randn(1, 7, 1024, 1024)
#
# # 前向传播
# output_tensor = model(input_tensor)
#
# # 打印输出张量的大小
# print("Output tensor size:", output_tensor.size())

