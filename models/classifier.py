from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lambd


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class ResClassifier(nn.Module):
    def __init__(self, num_classes=12, num_layer=2, num_unit=2048, prob=0.5, middle=1000):
        super(ResClassifier, self).__init__()
        layers1 = []
        # currently 10000 units
        layers1.append(nn.Dropout(p=prob))
        fc11 = nn.Linear(num_unit, middle)
        self.fc11 = fc11
        layers1.append(fc11)
        layers1.append(nn.BatchNorm1d(middle, affine=True))
        layers1.append(nn.ReLU(inplace=True))
        layers1.append(nn.Dropout(p=prob))
        fc12 = nn.Linear(middle, middle)
        self.fc12 = fc12
        layers1.append(fc12)
        layers1.append(nn.BatchNorm1d(middle, affine=True))
        layers1.append(nn.ReLU(inplace=True))
        fc13 = nn.Linear(middle, num_classes)
        self.fc13 = fc13
        layers1.append(fc13)
        self.classifier1 = nn.Sequential(*layers1)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        y = self.classifier1(x)
        return y
