from types import SimpleNamespace

import torch


def itemize(d):
    stat = {}
    for k, v in d.items():
        if type(v) == dict:
            stat[k] = itemize(v)
        else:
            stat[k] = v.item()
    return stat


class TensorBatch(SimpleNamespace):
    def __add__(self, other):
        return TensorBatch(**{
            k: torch.cat([v, other.__dict__[k]], dim=0)
            for k, v in self.__dict__.items() if type(v) == torch.Tensor
        })

class ToDeviceMixin:
    device = None

    def to(self, device):
        self.device = device

        parent = super(ToDeviceMixin, self)
        if hasattr(parent, 'to'):
            return parent.to(device)
        else:
            return self