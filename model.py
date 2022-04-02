from torch import nn


class HardSwish(nn.Module):  # 5.2 Nonlinearities
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):  # TODO
        return


class HardSigmoid(nn.Module):  # 5.2 Nonlinearsities
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):  # TODO
        return


class SqueezeExcite(nn.Module):  # TODO
    def __init__(self):  # 5.3 Large squeeze-and-excite, reduce to 1/4
        super(SqueezeExcite, self).__init__()

    def forward(self, x):
        return


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
