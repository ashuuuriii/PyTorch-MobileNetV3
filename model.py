from torch import nn
import torch.nn.functional as F


class HardSwish(nn.Module):  # 5.2 Nonlinearities
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x+3., inplace=self.inplace) * 1./6. * x


class HardSigmoid(nn.Module):  # 5.2 Nonlinearsities
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x+3., inplace=self.inplace) * 1./6.


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=4, inplace=True):  # 5.3 Large squeeze-and-excite, reduce to 1/4
        super(SqueezeExcite, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # ((N), C, output_size, output_size), reduces to ((N), C, 1, 1)
        self.excite = nn.Sequential(nn.Linear(c, c//r, bias=False),
                                    nn.ReLU(inplace),
                                    nn.Linear(c//r, c, bias=False),
                                    HardSigmoid(inplace))

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excite(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class InvertedBottleNeck(nn.Module):  # TODO
    def __init__(self):
        super(InvertedBottleNeck, self).__init__()

    def forward(self, x):
        return


class MobileNetV3(nn.Module):  # TODO
    def __init__(self):
        super(MobileNetV3, self).__init__()

    def forward(self, x):
        return
