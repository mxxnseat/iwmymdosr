from cordconv2d.model.model import Iwmymdosr
import torch as t

@t.no_grad
def evaluate(model: Iwmymdosr, inp: t.Tensor):
    model.eval()
    return model.forward(inp)
