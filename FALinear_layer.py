# Implementation of linear feedback alignment layer 
# as described in Lillicrap et al., 2016 (https://www.nature.com/articles/ncomms13276)
# Adapted from code example from https://pytorch.org/docs/stable/notes/extending.html
# and https://github.com/pytorch/pytorch/blob/aa2782b9ecf32cb19f1f613c75c16a7bc56f8533/torch/nn/modules/linear.py#L35

import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor

import math

## Feedback Alignment Linear
class FALinearFunc(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, weight_fb, bias=None):
        ctx.save_for_backward(input, weight, weight_fb, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, weight_fb, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_weight_fb = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_fb) # use weight_fb instead of weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_weight_fb, grad_bias


class FALinear(nn.Module):
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(FALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_fb', torch.Tensor(out_features, in_features)) # feedback weight
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_fb, a=math.sqrt(5)) # initialize feedback weight
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return FALinearFunc.apply(input, self.weight, self.weight_fb, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
