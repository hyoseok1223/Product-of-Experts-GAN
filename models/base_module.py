'''
    Based on https://github.com/rosinality/stylegan2-pytorch
'''

import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from math import sqrt

# from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix

def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)
        # clamp
        out = torch.clamp(out,min=-256,max=256)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4
        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)
        # clamp
        out = torch.clamp(out,min=-256,max=256)
        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    @custom_bwd
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply

class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        # clamp
        return torch.clamp(self.conv(input),min=-256,max=256)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        #clamp
        return torch.clamp(self.linear(input),min=-256,max=256)


# https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
class DownSample(nn.Module):
    def __init__(self, size, mode='nearest'):
        super(DownSample, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        downsample=False,
        upsample = False,
        fused=False,
    ):
        super().__init__()

        padding
        kernel_size


        if downsample:
            if fused:
                self.conv = nn.Sequential(
                    Blur(in_channel),
                    FusedDownsample(in_channel, out_channel, kernel_size, padding=padding),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv = nn.Sequential(
                    Blur(in_channel),
                    EqualConv2d(in_channel, out_channel, kernel_size, padding=padding),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        elif upsample:
            if fused:
                self.conv = nn.Sequential(
                    FusedUpsample(in_channel, out_channel, kernel_size, padding=padding),
                    Blur(out_channel),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    EqualConv2d( in_channel, out_channel, kernel_size, padding=padding),
                    Blur(out_channel),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv = nn.Sequential(
                EqualConv2d(in_channel, out_channel, kernel_size, padding=padding),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv(input)
        return out

class PoE(nn.Module):
    def __init__(self):
        super(PoE, self).__init__()

    def forward(self, mu_list, logvar_list, eps=1e-8):
        # mu : N x 
        T_sum = 0
        mu_T_sum = 0
        for mu, logvar in zip(mu_list,logvar_list):

            var = torch.exp(logvar)+ eps
            T = 1/ (var+eps)

            T_sum += T
            mu_T_sum += mu*T

        mu = mu_T_sum / T_sum
        var = 1/T_sum
        logvar = torch.log(var+eps)
        
        return mu, logvar

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3) # style정보를 N,C,1,1로 늘려줌.
        # W는 N,512,1,1로 늘려줌.
        # 저기에다 더해줄 거임. 저 out이 의미하는 바가 음.. 
        # intance normalization을 해주면,크기는 그대로임. 즉, feature map이라는 것임.
        # 절대 1,1아닌데 곱해지는게 가능 앞에 두개가 맞아서 그런가. 
        # 그러면 이거 그대로 쓰면 되게싿. 
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class ConstantInput(nn.Module):
    def __init__(self, channel, size):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch):
        out = self.input.repeat(batch, 1, 1, 1)

        return out