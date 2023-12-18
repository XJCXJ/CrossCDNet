# -*- coding: utf-8 -*-
"""
@Time    : 2023/3/28/028 13:57
@Author  : NDWX
@File    : SiamIBN.py
@Software: PyCharm
resnet18-ibn-1
"""
import torch
import torch.nn as nn
import math
from mmengine.model import BaseModule
# from mmseg.models.builder import BACKBONES
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from opencd.registry import MODELS



model_urls = {
    'resnet18_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_b-bc2f3c11.pth',
    'resnet34_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_b-04134c37.pth',
    'resnet50_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_b-9ca61e85.pth',
    'resnet101_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_b-c55f6dba.pth',
}


class AsymGlobalAttn(BaseModule):
    def __init__(self, dim, strip_kernel_size=21):
        super().__init__()

        self.norm = build_norm_layer(dict(type='mmpretrain.LN2d', eps=1e-6), dim)[1]
        self.global_ = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, (1, strip_kernel_size), padding=(0, (strip_kernel_size - 1) // 2), groups=dim),
            nn.Conv2d(dim, dim, (strip_kernel_size, 1), padding=((strip_kernel_size - 1) // 2, 0), groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.layer_scale = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        a = self.global_(x)
        x = a * self.v(x)
        x = self.proj(x)
        x = self.norm(x)
        x = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x + identity


        return x

class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class BasicBlock_IBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(BasicBlock_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.IN = nn.InstanceNorm2d(planes, affine=True) if ibn == 'b' else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 strip_kernel_size=(41, 31, 21, 11),
                 use_global=(True, True, True, True)):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
            # self.bn1 = nn.BatchNorm2d(64)
        else:
            # self.bn1 = nn.InstanceNorm2d(64, affine=True)
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0], strip_kernel_size[0], ibn=ibn_cfg[0],
                                       use_global=use_global[0])
        self.layer2 = self._make_layer(block, 128, layers[1], strip_kernel_size[1], stride=2, ibn=ibn_cfg[1],
                                       use_global=use_global[1])
        self.layer3 = self._make_layer(block, 256, layers[2], strip_kernel_size[2], stride=2, ibn=ibn_cfg[2],
                                       use_global=use_global[2])
        self.layer4 = self._make_layer(block, 512, layers[3], strip_kernel_size[3], stride=2, ibn=ibn_cfg[3],
                                       use_global=use_global[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, strip_kernel_size, stride=1, ibn=None, use_global=True,
                    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks - 1) else ibn))
            if use_global:
                layers.append(
                    AsymGlobalAttn(planes, strip_kernel_size)  # planes = 64
                )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.pool(x)
        # print(x.size())

        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)
        x = self.layer4(x)
        out.append(x)
        return out


@MODELS.register_module()
def resnet18_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-18-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[2, 2, 2, 2],
                       ibn_cfg=('b', 'b', None, None),
                       use_global=(True, True, True, True),
                       **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet18_ibn_b'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@MODELS.register_module()
def resnet34_ibn_b(pretrained=False, interaction_cfg=(None, None, None, None), **kwargs):
    """Constructs a ResNet-34-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet34_ibn_b'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@MODELS.register_module()
def resnet50_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-50-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_b'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@MODELS.register_module()
def resnet101_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('b', 'b', None, None),
                       # use_global =(False, False, False,False),
                       **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_b'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    model = resnet18_ibn_b(pretrained=False)
    x1 = torch.randn(4, 3, 256, 256)
    output = model(x1)
    # print(output)
    # for num, i in enumerate(output):
    #     print(num)
    #     print(i.size())
