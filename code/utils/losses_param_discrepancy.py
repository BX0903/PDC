import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cosdistance_loss_in(model_1, model_2):
    param_loss = 0.
    model_size = 0.
    for (param_1, param_2) in zip(model_1.parameters(), model_2.parameters()):
        param_1 = param_1.view(-1)
        param_2 = param_2.view(-1)

        param_loss = param_loss + (torch.dot(param_1, param_2) /
                                   (torch.sqrt(torch.sum(param_1 ** 2) + 1e-7) *
                                    torch.sqrt(torch.sum(param_2 ** 2) + 1e-7))) ** 2
        model_size += 1

    return param_loss / model_size


if __name__ == '__main__':
    a = torch.randn(1, 3, 64, 64, 64)
    b = torch.randn(1, 3, 64, 64, 64)

    a = F.softmax(a, dim=1)
    b = F.softmax(b, dim=1)
