#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
#尝试不同的group=2、4、8、16、32
#更改后的inception2
class Inception(nn.Module):  #Inception模块
    # ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5代表Inception对应卷积核的个数。
    def __init__(self, dim):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(dim, int(dim/4), kernel_size=3, padding=1, groups=8)
        self.branch2 = nn.Conv2d(dim, int(dim/4), kernel_size=5, padding=2, groups=8)
        self.branch3 = nn.Conv2d(dim, int(dim/4), kernel_size=7, padding=3, groups=8)
        self.branch4 = nn.Conv2d(dim, int(dim/4), kernel_size=9, padding=4, groups=8)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)  #[N,C,H,W].在C通道上拼接。torch.cat对tensor进行拼接





