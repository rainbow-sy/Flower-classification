import torch.nn as nn
import torch

class Involution(nn.Module):  #不改变通道数大小，若stride=1，图像H、W不变；若stride=2，图像H、W变为原来的一半
    def __init__(self, in_channel, kernel_size=7, stride=1):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = in_channel
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        # -----------------------------------------
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // reduction_ratio,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel // reduction_ratio)
        self.relu = nn.ReLU()
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=in_channel // reduction_ratio, out_channels=kernel_size ** 2 * self.groups,
                               kernel_size=1, stride=1)
        # -----------------------------------------
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        # -----------------------------------------
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=1, padding = (kernel_size - 1) // 2, stride=stride)

    def forward(self, x):
        weight = self.conv1(x if self.stride == 1 else self.avgpool(x))
        weight = self.bn1(weight)
        weight = self.relu(weight)
        weight = self.conv2(weight)
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out

def main():
    batches_img = torch.rand(1, 128, 56, 56)
    involuton=Involution(in_channel=128,kernel_size=5,stride=4)
    out =involuton(batches_img)
    print(out.shape)

if __name__ == '__main__':
    main()
