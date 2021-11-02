

import paddle
import paddle.nn as nn 
import paddle.nn.functional as F
from .resnet import Resnet18
from reprod_log import ReprodLogger
from paddle.nn import BatchNorm2D
# reprod_logger = ReprodLogger()


class ConvBNReLU(paddle.nn.Layer):

    def __init__(self, in_chan, out_chan, ks, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2D(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias_attr=False)
        self.bn = BatchNorm2D(out_chan,momentum=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x





class BiSeNetOutput(paddle.nn.Layer):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, 1, 1)
        self.conv_out = nn.Conv2D(mid_chan, out_chan, kernel_size=1)
        self.up = nn.Upsample(scale_factor=up_factor,
                mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        # reprod_logger.add("x1", x.numpy())
        x = self.conv_out(x)
        # reprod_logger.add("x2", x.numpy())
        x = self.up(x)
        
        return x


    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_sublayers():
            print(name)
            if isinstance(module, nn.Conv2D):
                print("gggggggggg",module)
                wd_params.append(module.weight)
                if name == "conv_out":
                    print("dddddddd")
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2D):
                # print(module.parameters())
                print("nn.bh")
                nowd_params += list(module.parameters())[:2]
                # print(nowd_params.shape)
        return wd_params, nowd_params


class AttentionRefinementModule(paddle.nn.Layer):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2D(out_chan, out_chan, kernel_size=1,bias_attr=False)
        self.bn_atten = BatchNorm2D(out_chan,momentum=0.1)
        self.sigmoid_atten = nn.Sigmoid()


    def forward(self, x):
        feat = self.conv(x)
        atten = paddle.mean(feat, axis=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        # atten = nn.Sigmoid(atten)
        out = paddle.multiply(feat, atten)
        return out




class ContextPath(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.)
        self.up16 = nn.Upsample(scale_factor=2.)


    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        print(feat8.shape, feat16.shape, feat32.shape)

        avg = paddle.mean(feat32, axis=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up # x8, x16



    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_sublayers():
            print(name)
            if isinstance(module, (nn.Conv2D)):
                print("gggggggggg",module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    print("dddddddd")
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2D):
                print("nn.bh")
                nowd_params += list(module.parameters())[:2]
                # print(nowd_params.dtype)
        return wd_params, nowd_params


class SpatialPath(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_sublayers():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2D):
                print("gggggggggg",module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    print("dddddddd")
                    nowd_params.append(module.bias)
                    print('module.bias:', len(nowd_params))
            elif isinstance(module, nn.BatchNorm2D):
                nowd_params += list(module.parameters())[:2]
                print("nn.bh:", list(module.parameters()))
        return wd_params, nowd_params


class FeatureFusionModule(paddle.nn.Layer):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv = nn.Conv2D(out_chan,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias_attr=False)
        self.bn = BatchNorm2D(out_chan,momentum=0.1)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = paddle.concat([fsp, fcp], axis=1)
        feat = self.convblk(fcat)
        atten = paddle.mean(feat, axis=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = self.sigmoid_atten(atten)
        feat_atten = paddle.multiply(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_sublayers():
            if isinstance(module, (nn.Conv2D)):
                print("gggggggggg",module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    print("module.bias.sum:",paddle.sum(module.bias))
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2D):
                print("nn.bh")
                nowd_params += list(module.parameters())[:2]
        return wd_params, nowd_params


class BiSeNetV1(paddle.nn.Layer):

    def __init__(self, n_classes, aux_mode='train', *args, **kwargs):
        super(BiSeNetV1, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)
        self.aux_mode = aux_mode
        if self.aux_mode == 'train':
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes, up_factor=8)
            self.conv_out32 = BiSeNetOutput(128, 64, n_classes, up_factor=16)


    def forward(self, x):

        feat_cp8, feat_cp16 = self.cp(x)
        # print(feat_cp8.shape, feat_cp16.shape)
        
        feat_sp = self.sp(x)
        # print(feat_sp.shape)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        # reprod_logger.add("feat_fuse", feat_fuse.numpy())
        # reprod_logger.add("feat_out", feat_out.numpy())
        # print(feat_out.shape)
        if self.aux_mode == 'train':
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)
            # reprod_logger.add("feat_out16", feat_out16.numpy())
            # reprod_logger.add("feat_out32", feat_out32.numpy())
            # reprod_logger.save("./pipeline/step_1/forward_paddle.npy")
            return feat_out, feat_out16, feat_out32
        elif self.aux_mode == 'eval':
            return feat_out,
        elif self.aux_mode == 'pred':
            feat_out = feat_out.argmax(dim=1)
            return feat_out
        else:
            raise NotImplementedError


    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            print("-------------",name)
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        # print(len(wd_params), len(nowd_params), len(lr_mul_wd_params), len(lr_mul_nowd_params))

        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    net = BiSeNetV1(19)
    net.eval()
    # print(net)
    in_ten = paddle.randn([1, 3, 1024, 1024])
    print(in_ten.shape)
    # load_layer_state_dict = paddle.load('./model_final_v1_city_new.pdparams')
    
    # net.set_dict(load_layer_state_dict)
    # # net.load_state_dict(paddle.load('./model_final_v1_city_new.pdparams', map_location='cpu'))
    out, out16, out32 = net(in_ten)
    # print(out.shape)
    # print(out16.shape)
    # print(out32.shape)

    # net.get_params()
    # # print(net)
