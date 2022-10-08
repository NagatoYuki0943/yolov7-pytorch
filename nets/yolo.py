import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, Multi_Concat_Block, Conv, SiLU, Transition_Block, autopad


#-----------------------------------------------#
#   SPP
#                   in
#                   │
#          ┌──────────────────┐
#       cv1(1x1)           cv2(1x1)
#          │                  │
#       cv3(3x3)              │
#          │                  │
#       cv4(1x1)              │
#          │                  │
#   MaxPool2d k=5,9,13        │
#          │                  │
#        concat               │
#          │                  │
#       cv5(1x1)              │
#          │                  │
#       cv6(3x3)              │
#          └────────┬─────────┘
#                concat
#                   │
#                cv7(1X1)
#                   │
#                  out
#-----------------------------------------------#
class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)

        # final
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

#-----------------------------------------------#
#   RepVGG的Conv
#   PANet的三个输出都加了一个，通道发生变化，因此没有使用rbr_identity
#-----------------------------------------------#
class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy         = deploy
        self.groups         = g
        self.in_channels    = c1
        self.out_channels   = c2

        assert k == 3
        assert autopad(k, p) == 1

        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        # 部署时使用合并后的3x3卷积
        if deploy:
            self.rbr_reparam    = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        # 训练时使用identity，3x3卷积和1x1卷积
        else:
            # identity只在宽高和通道都不变时才使用
            self.rbr_identity   = (nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense      = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            # 1x1卷积padding=0
            padding_11  = autopad(k, p) - k // 2
            self.rbr_1x1        = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        # 部署时使用合并后的3x3卷积
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        # 不使用identity时直接加0
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    #-------------------------------------------#
    #   没用上
    #   合并3条分支的卷积和bn，返回kernel和bias
    #-------------------------------------------#
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3  = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1  = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid    = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    #--------------------------------------------#
    #   没用上
    #   1x1conv填充为3x3conv
    #--------------------------------------------#
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    #--------------------------------------------#
    #   没用上
    #   合并1条分支的卷积和bn，返回kernel和bias
    #--------------------------------------------#
    def _fuse_bn_tensor(self, branch):
        # rbr_identity分支在形状变化时为None
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel      = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma       = branch[1].weight
            beta        = branch[1].bias
            eps         = branch[1].eps
        else:
            # identity分支只有一个bn
            assert isinstance(branch, nn.BatchNorm2d)
            # 创建中心为1，周围为0的3x3卷积核，这样经过卷积后值不变
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel      = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma       = branch.weight
            beta        = branch.bias
            eps         = branch.eps
        std = (running_var + eps).sqrt()            # 标准差
        t   = (gamma / std).reshape(-1, 1, 1, 1)    # \frac{\gamma}{\sqrt{var}}  gamma/std
        return kernel * t, beta - running_mean * gamma / std

    #--------------------------------------------#
    #   没用上
    #--------------------------------------------#
    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std     = (bn.running_var + bn.eps).sqrt()              # 标准差
        bias    = bn.bias - bn.running_mean * bn.weight / std   # 偏置

        t       = (bn.weight / std).reshape(-1, 1, 1, 1)        # \frac{\gamma}{\sqrt{var}}  gamma/std
        weights = conv.weight * t

        bn      = nn.Identity()
        conv    = nn.Conv2d(in_channels = conv.in_channels,
                            out_channels = conv.out_channels,
                            kernel_size = conv.kernel_size,
                            stride=conv.stride,
                            padding = conv.padding,
                            dilation = conv.dilation,
                            groups = conv.groups,
                            bias = True,
                            padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias   = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        # 3x3
        self.rbr_dense  = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        # 1x1
        self.rbr_1x1    = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias    = self.rbr_1x1.bias
        # rbr_identity
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups,
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1           = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded      = identity_conv_1x1.bias
            weight_identity_expanded    = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded      = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded    = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )

        self.rbr_dense.weight   = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias     = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam    = self.rbr_dense
        self.deploy         = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None

def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv  = conv.weight.clone().view(conv.out_channels, -1)
    w_bn    = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv  = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn    = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        #-----------------------------------------------#
        #   定义了不同yolov7版本的参数
        #-----------------------------------------------#
        transition_channels = {'l' : 32, 'x' : 40}[phi]
        block_channels      = 32
        panet_channels      = {'l' : 32, 'x' : 64}[phi]
        e       = {'l' : 2, 'x' : 1}[phi]
        n       = {'l' : 4, 'x' : 6}[phi]
        ids     = {'l' : [-1, -2, -3, -4, -5, -6], 'x' : [-1, -3, -5, -7, -8]}[phi]
        conv    = {'l' : RepConv, 'x' : Conv}[phi]
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#

        #---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   1, 512,  80, 80
        #   1, 1024, 40, 40
        #   1, 1024, 20, 20
        #---------------------------------------------------#
        self.backbone   = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained)

        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        #---------------------------------------------------#
        #   feat3的spp模块，通道减半
        #   [1, 1024, 20, 20] -> [1, 512, 20, 20]
        #---------------------------------------------------#
        self.sppcspc                = SPPCSPC(transition_channels * 32, transition_channels * 16)

        #---------------------------------------------------#
        #   PANet上采样部分
        #---------------------------------------------------#
        #   [1, 512, 20, 20] -> [1, 256, 20, 20]
        self.conv_for_P5            = Conv(transition_channels * 16, transition_channels * 8)
        #   [1, 1024,40, 40] -> [1, 256, 40, 40]
        self.conv_for_feat2         = Conv(transition_channels * 32, transition_channels * 8)
        #---------------------------------------------------#
        #   backbone的dark block 参数2是隐藏通道,参数3是输出通道
        #   [1, 512, 40, 40] -> [1, 256, 40, 40]
        #---------------------------------------------------#
        self.conv3_for_upsample1    = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        #   [1, 256, 40, 40] -> [1, 128, 40, 40]
        self.conv_for_P4            = Conv(transition_channels * 8, transition_channels * 4)
        #   [1, 512, 80, 80] -> [1, 128, 80, 80]
        self.conv_for_feat1         = Conv(transition_channels * 16, transition_channels * 4)
        #---------------------------------------------------#
        #   backbone的dark block
        #   [1, 256, 80, 80] -> [1, 128, 80, 80]
        #---------------------------------------------------#
        self.conv3_for_upsample2    = Multi_Concat_Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids)

        #---------------------------------------------------#
        #   PANet下采样部分
        #   下采样使用了backbone的Transition,因为参数2是两个分支独自的out_channel且会拼接,所以最终通道翻倍
        #---------------------------------------------------#
        #   [1, 128, 80, 80] -> [1, 256, 40, 40]
        self.down_sample1           = Transition_Block(transition_channels * 4, transition_channels * 4)
        #---------------------------------------------------#
        #   backbone的dark block
        #   [1, 512, 40, 40] -> [1, 256, 40, 40]
        #---------------------------------------------------#
        self.conv3_for_downsample1  = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        #   [1, 256, 40, 40] -> [1, 512, 20, 20]
        self.down_sample2           = Transition_Block(transition_channels * 8, transition_channels * 8)
        #---------------------------------------------------#
        #   backbone的dark block
        #   [1,1024, 20, 20] -> [1, 512, 20, 20]
        #---------------------------------------------------#
        self.conv3_for_downsample2  = Multi_Concat_Block(transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids)

        #---------------------------------------------------#
        #   repvgg部分,对PANet得3个输出进行计算
        #---------------------------------------------------#
        #   [1, 128, 80, 80] -> [1, 256, 80, 80]
        self.rep_conv_1 = conv(transition_channels * 4, transition_channels * 8, 3, 1)
        #   [1, 256, 40, 40] -> [1, 512, 40, 40]
        self.rep_conv_2 = conv(transition_channels * 8, transition_channels * 16, 3, 1)
        #   [1, 512, 20, 20] -> [1,1024, 20, 20]
        self.rep_conv_3 = conv(transition_channels * 16, transition_channels * 32, 3, 1)

        #---------------------------------------------------#
        #   对repvgg的三个输出进行计算三个特征层
        #   y3 = [1, 256, 80, 80] -> [1, 3*(num_classes+4+1), 80, 80]
        #   y2 = [1, 512, 40, 40] -> [1, 3*(num_classes+4+1), 40, 40]
        #   y1 = [1,1024, 20, 20] -> [b, 3*(num_classes+4+1), 20, 20]
        #---------------------------------------------------#
        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self

    def forward(self, x):
        #---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   1, 512,  80, 80
        #   1, 1024, 40, 40
        #   1, 1024, 20, 20
        #---------------------------------------------------#
        feat1, feat2, feat3 = self.backbone.forward(x)

        #---------------------------------------------------#
        #   feat3的spp模块，通道减半
        #   [1, 1024, 20, 20] -> [1, 512, 20, 20]
        #---------------------------------------------------#
        P5          = self.sppcspc(feat3)
        #---------------------------------------------------#
        #   PANet上采样部分
        #---------------------------------------------------#
        P5_conv     = self.conv_for_P5(P5)                                      # [1, 512, 20, 20] -> [1, 256, 20, 20]
        P5_upsample = self.upsample(P5_conv)                                    # [1, 256, 20, 20] -> [1, 256, 40, 40]
        P4          = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)   # ([1,1024,40, 40] -> [1, 256, 40, 40]) cat [1, 256, 40, 40] = [1, 512, 40, 40]
        P4_td       = self.conv3_for_upsample1(P4)                              # [1, 512, 40, 40] -> [1, 256, 40, 40]

        P4_conv     = self.conv_for_P4(P4_td)                                   # [1, 256, 40, 40] -> [1, 128, 40, 40]
        P4_upsample = self.upsample(P4_conv)                                    # [1, 128, 40, 40] -> [1, 128, 80, 80]
        P3          = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)   # ([1,512, 80, 80] -> [1, 128, 80, 80]) cat [1, 128, 80, 80] = [1, 256, 80, 80]
        P3_out      = self.conv3_for_upsample2(P3)                              # [1, 256, 80, 80] -> [1, 128, 80, 80]

        #---------------------------------------------------#
        #   PANet下采样部分
        #---------------------------------------------------#
        P3_downsample = self.down_sample1(P3_out)                               # [1, 128, 80, 80] -> [1, 256, 40, 40]
        P4_td         = torch.cat([P3_downsample, P4_td], 1)                    # [1, 256, 40, 40]cat [1, 256, 40, 40] = [1, 512, 40, 40]
        P4_out        = self.conv3_for_downsample1(P4_td)                       # [1, 512, 40, 40] -> [1, 256, 40, 40]

        P4_downsample = self.down_sample2(P4_out)                               # [1, 256, 40, 40] -> [1, 512, 20, 20]
        P5            = torch.cat([P4_downsample, P5], 1)                       # [1, 512, 20, 20]cat [1, 512, 20, 20] = [1, 1024, 20, 20]
        P5_out        = self.conv3_for_downsample2(P5)                          # [1,1024, 20, 20] -> [1, 512, 20, 20]

        #---------------------------------------------------#
        #   repvgg部分,对PANet得3个输出进行计算
        #---------------------------------------------------#
        P3_rep = self.rep_conv_1(P3_out)                                        # [1, 128, 80, 80] -> [1, 256, 80, 80]
        P4_rep = self.rep_conv_2(P4_out)                                        # [1, 256, 40, 40] -> [1, 512, 40, 40]
        P5_rep = self.rep_conv_3(P5_out)                                        # [1, 512, 20, 20] -> [1,1024, 20, 20]

        #---------------------------------------------------#
        #   第三个特征层
        #   y3 = [1, 256, 80, 80] -> [1, 3*(num_classes+4+1), 80, 80]
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3_rep)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2 = [1, 512, 40, 40] -> [1, 3*(num_classes+4+1), 40, 40]
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4_rep)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1 = [1,1024, 20, 20] -> [b, 3*(num_classes+4+1), 20, 20]
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5_rep)

        return [out0, out1, out2]


if __name__ == "__main__":
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes  = 20
    phi          = 'l'
    yolobody = YoloBody(anchors_mask, num_classes, phi)
    x = torch.randn(1, 3, 640, 640)
    yolobody.fuse();

    yolobody.eval()
    out0, out1, out2 = yolobody(x)
    print(out0.size())     # [1, 75, 20, 20]
    print(out1.size())     # [1, 75, 40, 40]
    print(out2.size())     # [1, 75, 80, 80]

    onnx_path = "./model_data/yolov7_weights_fuse.onnx"
    torch.onnx.export(yolobody,                     # 保存的模型
                        x,                          # 模型输入
                        onnx_path,                  # 模型保存 (can be a file or file-like object)
                        export_params=True,         # 如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
                        verbose=False,              # 如果为True，则打印一些转换日志，并且onnx模型中会包含doc_string信息
                        opset_version=15,           # ONNX version 值必须等于_onnx_main_opset或在_onnx_stable_opsets之内。具体可在torch/onnx/symbolic_helper.py中找到
                        do_constant_folding=True,   # 是否使用“常量折叠”优化。常量折叠将使用一些算好的常量来优化一些输入全为常量的节点。
                        input_names=["input"],      # 按顺序分配给onnx图的输入节点的名称列表
                        output_names=["out0", "out1", "out2"],    # 按顺序分配给onnx图的输出节点的名称列表
                        dynamic_axes={"input": {0: "batch_size"},   # 动态
                                      "output": {0: "batch_size"}})

    import onnx
    from onnxsim import simplify
    # 载入onnx模块
    model_ = onnx.load(onnx_path)
    # print(model_)

    # 简化模型,更好看
    model_simp, check = simplify(model_)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    print('finished exporting onnx')

    # 检查IR是否良好
    try:
        onnx.checker.check_model(model_)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")
