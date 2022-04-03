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


def calc_pad(k: int) -> int:
    return (k - 1) // 2


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


class InvertedBottleNeck(nn.Module):
    def __init__(self,
                 ic: int,
                 oc: int,
                 k: int,
                 exp: int,
                 s: int,
                 se: bool,
                 act: str):
        super(InvertedBottleNeck, self).__init__()
        self.pw1 = nn.Sequential(nn.Conv2d(ic, exp, 1, bias=False),
                                 nn.BatchNorm2d(exp))
        self.dw = nn.Sequential(nn.Conv2d(exp, exp, k, stride=s, groups=exp, padding=calc_pad(k), bias=False),
                                nn.BatchNorm2d(exp))
        self.pw2 = nn.Sequential(nn.Conv2d(exp, oc, 1, bias=False),
                                 nn.BatchNorm2d(oc))
        self.se = SqueezeExcite(exp) if se else None

        if act == 'RE':
            self.act = nn.ReLU(inplace=True)
        elif act == 'HS':
            self.act = HardSwish(inplace=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        residual = x
        y = self.pw1(x)
        y = self.act(y)
        y = self.dw(y)
        y = self.act(y)

        if self.se:
            y = self.se(y)

        y = self.pw2(y)

        if self.drop_out:
            y = self.drop_out(y)

        return y + residual


class Classifier(nn.Module):
    def __init__(self, head_type: str, last_in: int, last_out: int, n_classes: int):
        super(Classifier, self).__init__()
        self.head_type = head_type
        if self.head_type == 'fc':
            self.classification_head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                     nn.Linear(last_out, n_classes),
                                                     nn.Softmax(1))
        elif self.head_type == 'conv':
            self.classification_head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                     nn.Conv2d(last_in, last_out, 1, stride=1),
                                                     HardSwish(inplace=True),
                                                     nn.Conv2d(last_out, n_classes, 1))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = x.mean(3).mean(2) if self.head_type == 'fc' else x
        y = self.classification_head(x)

        if self.head_type == 'conv':
            return y.view(y.shape[0], -1)
        else:
            return y


class MobileNetV3(nn.Module):  # TODO
    def __init__(self, model_size: str, n_classes: int, head_type: str):
        assert model_size == 'small' or model_size == 'large',\
            "model_size should be 'small' or 'large'."

        super(MobileNetV3, self).__init__()

        if model_size == 'small':
            last_in = 576
            last_out = 1024
        elif model_size == 'large':
            last_in = 960
            last_out = 1280

        self.model = self._build_model(model_size, last_in)
        self.classifier = Classifier(head_type, last_in, last_out, n_classes)

    def forward(self, x):
        y = self.model(x)
        y = self.classifier(y)
        return y

    def _build_model(self, model_size: str, last_in: int) -> nn.Sequential:
        modules = nn.Sequential()
        ic = 16

        # Build the first block.
        block_0 = nn.Sequential(nn.Conv2d(3, ic, 3, stride=2, padding=calc_pad(3), bias=False),
                                nn.BatchNorm2d(ic),
                                HardSwish(inplace=True))
        modules.append(block_0)

        # Build the MobileNet bottleneck blocks.
        defs = module_defs[model_size]
        for bn in defs:
            modules.append(InvertedBottleNeck(ic, bn['oc'], bn['k'], bn['exp'], bn['s'], bn['se'], bn['act']))
            ic = bn['oc']

        # Build the last few blocks.
        modules.append(nn.Conv2d(ic, last_in, 1, stride=1, bias=False))
        modules.append(nn.BatchNorm2d(last_in))
        modules.append(HardSwish(inplace=True))

        return modules
