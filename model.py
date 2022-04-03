from torch import nn
import torch.nn.functional as F


module_defs = {
    # Table 1. and 2.
    # kernel, exp size, out channels, squeeze excite, nonlinearities, stride
    'large': [{'k': 3, 'exp': 16, 'oc': 16, 'se': False, 'act': 'RE', 's': 1},
              {'k': 3, 'exp': 64, 'oc': 24, 'se': False, 'act': 'RE', 's': 2},
              {'k': 3, 'exp': 72, 'oc': 24, 'se': False, 'act': 'RE', 's': 1},
              {'k': 5, 'exp': 72, 'oc': 40, 'se': True, 'act': 'RE', 's': 2},
              {'k': 5, 'exp': 120, 'oc': 40, 'se': True, 'act': 'RE', 's': 1},
              {'k': 5, 'exp': 120, 'oc': 40, 'se': True, 'act': 'RE', 's': 1},
              {'k': 3, 'exp': 240, 'oc': 80, 'se': False, 'act': 'HS', 's': 2},
              {'k': 3, 'exp': 200, 'oc': 80, 'se': False, 'act': 'HS', 's': 1},
              {'k': 3, 'exp': 184, 'oc': 80, 'se': False, 'act': 'HS', 's': 1},
              {'k': 3, 'exp': 184, 'oc': 80, 'se': False, 'act': 'HS', 's': 1},
              {'k': 3, 'exp': 480, 'oc': 112, 'se': True, 'act': 'HS', 's': 1},
              {'k': 3, 'exp': 672, 'oc': 112, 'se': True, 'act': 'HS', 's': 1},
              {'k': 5, 'exp': 672, 'oc': 160, 'se': True, 'act': 'HS', 's': 2},
              {'k': 5, 'exp': 960, 'oc': 160, 'se': True, 'act': 'HS', 's': 1},
              {'k': 5, 'exp': 960, 'oc': 160, 'se': True, 'act': 'HS', 's': 1}],
    'small': [{'k': 3, 'exp': 16, 'oc': 16, 'se': True, 'act': 'RE', 's': 2},
              {'k': 3, 'exp': 72, 'oc': 24, 'se': False, 'act': 'RE', 's': 2},
              {'k': 3, 'exp': 88, 'oc': 24, 'se': False, 'act': 'RE', 's': 1},
              {'k': 5, 'exp': 96, 'oc': 40, 'se': True, 'act': 'HS', 's': 2},
              {'k': 5, 'exp': 240, 'oc': 40, 'se': True, 'act': 'HS', 's': 1},
              {'k': 5, 'exp': 240, 'oc': 240, 'se': True, 'act': 'HS', 's': 1},
              {'k': 5, 'exp': 120, 'oc': 48, 'se': True, 'act': 'HS', 's': 1},
              {'k': 5, 'exp': 144, 'oc': 48, 'se': True, 'act': 'HS', 's': 1},
              {'k': 5, 'exp': 288, 'oc': 96, 'se': True, 'act': 'HS', 's': 2},
              {'k': 5, 'exp': 576, 'oc': 96, 'se': True, 'act': 'HS', 's': 1},
              {'k': 5, 'exp': 576, 'oc': 96, 'se': True, 'act': 'HS', 's': 1}]
}


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
    def __init__(self, model_size: str):
        assert model_size == 'small' or model_size == 'large',\
            "model_size should be 'small' or 'large'."

        super(MobileNetV3, self).__init__()

        if model_size == 'small':
            last_in = 576
            last_out = 1024
        elif model_size == 'large':
            last_in = 960
            last_out = 1280

        self.model = self._build_model(model_size)

    def forward(self, x):
        return

    def _build_model(self, model_size:str) -> nn.Sequential:
        modules = nn.Sequential()

        return modules
