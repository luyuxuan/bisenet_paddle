#!/usr/bin/python
# -*- encoding: utf-8 -*-

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.model_zoo as modelzoo

# from torch.nn import BatchNorm2d

import paddle.nn.functional as F
import paddle
import paddle.nn as nn
# paddle.nn.Conv2d
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1,bias_attr=False)


class BasicBlock(paddle.nn.Layer):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2D(out_chan,momentum=0.1)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2D(out_chan,momentum=0.1)
        self.relu = nn.ReLU()
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_chan, out_chan,
                          kernel_size=1, stride=stride,bias_attr=False),
                nn.BatchNorm2D(out_chan,momentum=0.1),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(paddle.nn.Layer):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3,bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64,momentum=0.1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        # self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat8, feat16, feat32

    # def init_weight(self):
    #     state_dict = modelzoo.load_url(resnet18_url)
    #     self_state_dict = self.state_dict()
    #     for k, v in state_dict.items():
    #         if 'fc' in k: continue
    #         self_state_dict.update({k: v})
    #     self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        print("meiguo _________________renet")
        for name, module in self.named_sublayers():
            print("-----------",name)
            if isinstance(module, (nn.Conv2D)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2D):
                # print(module.dtype)
                print("nn.bh")
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


if __name__ == "__main__":
    net = Resnet18()
    x = paddle.randn([16,3,224,224])
    # x = torch.randn(16, 3, 224, 224)
    out = net(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    net.get_params()
    print(net.get_params())
