import torch
import torch.nn as nn


#-----------------#
#   Conv的padding
#-----------------#
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


#-----------------#
#   Conv+BN+SiLU
#-----------------#
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    #---------#
    #   训练
    #---------#
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    #-------------------------#
    #   推理过程,bn融合进卷积
    #-------------------------#
    def fuseforward(self, x):
        return self.act(self.conv(x))


#-----------------------------------------------#
#   dark2,dark3,dark4,dark5的block
#                   in
#                   │
#          ┌────────┤
#       cv1(1x1) cv2(1x1)
#          │        ├─────────┐
#          │        │     cv3(3x3)_1
#          │        │         │
#          │        │     cv3(3x3)_2
#          │        │         ├─────────┐
#          │        │         │     cv3(3x3)_3
#          │        │         │         │
#          │        │         │     cv3(3x3)_4
#          │        │         │         ├ ─ ─ ─ ─ ┐
#          │        │         │         │      cv3(3x3)_N cv3会有多个卷积,多个输出
#          └────────┼─────────┴─────────┘─ ─ ─ ─ ─┘
#                concat
#                   │
#                cv4(1X1)
#                   │
#                  out
#-----------------------------------------------#
class Multi_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)

        self.ids = ids
        #-----------------------------------------------#
        #   cv1,cv2和cv3的全部输出都会被保留下来
        #   cv1是一个单独输出,类似于CSP的短接分支
        #   cv2和cv3是串联在一起的
        #-----------------------------------------------#
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)]
        )
        #-----------------------------------------------#
        #   cv1和cv2以及cv3的输出会被返回
        #   len(ids) - 2 是因为ids包括了cv1和cv2的输出
        #-----------------------------------------------#
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)

        x_all = [x_1, x_2]
        # [-1, -3, -5, -6] => [5, 3, 1, 0]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
        #-----------------------------------------------#
        #   cv1和cv2以及cv3的输出会被返回
        #-----------------------------------------------#
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out


#-----------------------------------------------#
#   MaxPool2d k=2 s=2
#-----------------------------------------------#
class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


#-----------------------------------------------#
#   dark3,dark4,dark5的下采样部分, 通道不变,宽高减半(针对backbone)
#   分支1 MaxPool2d + Conv
#   分支2 Conv      + Conv
#   最后将2个分支拼接返回
#-----------------------------------------------#
class Transition_Block(nn.Module):
    def __init__(self, c1, c2):
        super(Transition_Block, self).__init__()

        self.mp  = MP()
        self.cv1 = Conv(c1, c2, 1, 1)
        # 分支2,通道减半(针对backbone)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)   # s=2

    def forward(self, x):
        # 160, 160, 256 => 80, 80, 256 => 80, 80, 128
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)

        # 160, 160, 256 => 160, 160, 128 => 80, 80, 128
        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)

        # 两个通道减半拼接到一起最终通道不变
        # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
        return torch.cat([x_2, x_1], 1)


class Backbone(nn.Module):
    def __init__(self, transition_channels, block_channels, n, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#
        #-----------------------------------------------#
        #   dark中的Block中的返回id数
        #   最后两个代表cv1和cv2的输出,前2(3)个代表cv3的返回
        #-----------------------------------------------#
        ids = {
            'l' : [-1, -3, -5, -6],
            'x' : [-1, -3, -5, -7, -8],
        }[phi]

        # [1, 3, 640, 640] -> [1, 64, 320, 320]
        self.stem = nn.Sequential(
            Conv(3, transition_channels, 3, 1),
            Conv(transition_channels, transition_channels * 2, 3, 2),       # s=2
            Conv(transition_channels * 2, transition_channels * 2, 3, 1),
        )

        # [1, 64, 320, 320] -> [1, 128, 160, 160] -> [1, 256, 160, 160]
        self.dark2 = nn.Sequential(
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),   # s=2
            Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids),
        )

        # [1, 256, 160, 160] -> [1, 256, 80, 80] -> [1, 512, 80, 80]
        # Transition这里的参数2是两个分支的输出channel,最后拼接到一起通道和输入时相同
        self.dark3 = nn.Sequential(
            Transition_Block(transition_channels * 8, transition_channels * 4),
            Multi_Concat_Block(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids),
        )

        # [1, 512, 80, 80] -> [1, 512, 40, 40] -> [1, 1024, 40, 40]
        self.dark4 = nn.Sequential(
            Transition_Block(transition_channels * 16, transition_channels * 8),
            Multi_Concat_Block(transition_channels * 16, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        )

        # [1, 1024, 40, 40] -> [1, 1024, 20, 20] -> [1, 1024, 20, 20]
        self.dark5 = nn.Sequential(
            Transition_Block(transition_channels * 32, transition_channels * 16),
            Multi_Concat_Block(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        )

        if pretrained:
            url = {
                "l" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        #-----------------------------------------------#
        #   [1, 3, 640, 640] -> [1, 64, 320, 320]
        #-----------------------------------------------#
        x = self.stem(x)
        #-----------------------------------------------#
        #   [1, 64, 320, 320] -> [1, 256, 160, 160]
        #-----------------------------------------------#
        x = self.dark2(x)
        #-----------------------------------------------#
        #   [1, 256, 160, 160] -> [1, 512, 80, 80]
        #   dark3的输出为80, 80, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   [1, 512, 80, 80] -> [1, 1024, 40, 40]
        #   dark4的输出为40, 40, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   [1, 1024, 40, 40] -> [1, 1024, 20, 20]
        #   dark5的输出，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3


if __name__ == "__main__":
    phi                 = 'l'
    transition_channels = {'l' : 32, 'x' : 40}[phi]
    block_channels      = 32
    n                   = {'l' : 4, 'x' : 6}[phi]
    pretrained          = False
    backbone = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained)
    x = torch.randn(1, 3, 640, 640)
    backbone.eval()
    feat1, feat2, feat3 = backbone(x)
    print(feat1.size())     # [1,  512, 80, 80]
    print(feat2.size())     # [1, 1024, 40, 40]
    print(feat3.size())     # [1, 1024, 20, 20]

    if False:
        onnx_path = "./model_data/yolov7_backbone.onnx"
        torch.onnx.export(backbone,                     # 保存的模型
                            x,                          # 模型输入
                            onnx_path,                  # 模型保存 (can be a file or file-like object)
                            export_params=True,         # 如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
                            verbose=False,              # 如果为True，则打印一些转换日志，并且onnx模型中会包含doc_string信息
                            opset_version=15,           # ONNX version 值必须等于_onnx_main_opset或在_onnx_stable_opsets之内。具体可在torch/onnx/symbolic_helper.py中找到
                            do_constant_folding=True,   # 是否使用“常量折叠”优化。常量折叠将使用一些算好的常量来优化一些输入全为常量的节点。
                            input_names=["input"],      # 按顺序分配给onnx图的输入节点的名称列表
                            output_names=["feat1", "feat2", "feat3"],    # 按顺序分配给onnx图的输出节点的名称列表
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

        # Transition 通道不变,宽高减半
        # trans = Transition(64, 32)
        # x = torch.randn(1, 64, 256, 256)
        # y = trans(x)
        # print(y.size()) # [1, 64, 128, 128]