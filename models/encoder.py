import torch.nn as nn
import numpy as np
import torch
import clip
from PIL import Image
from typing import *

from models.base_module import *
from torchvision.transforms.functional import to_pil_image

# a 4-layer MLP with dimension 512 that processes the
# CLIP embedding of a caption


config_input_shape = (256,256) # (1024,1024)
config_stage = 3 # 5
config_spatial_channel = 64 # 32
config_init_channel = 64
config_encoder_init_channel = 16//2

class TextEncoder(nn.Module):
    def __init__(self,
                 hidden_dims: List = None
                  ):
        super(TextEncoder, self).__init__()

        # Build Encoder
        modules = []
        if hidden_dims is None:
            hidden_dims = [512, 512, 512, 512]
            
        in_channels = 512 # clip output channel size is 512
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    EqualLinear(in_channels, h_dim),
                    # nn.BatchNorm2d(h_dim, affine=True), 
                    nn.LeakyReLU(0.2))
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.out_dim = in_channels

        self.clip, _ = clip.load('ViT-B/32') 
        resolution = self.clip.visual.input_resolution
        self.preprocess = DownSample((resolution,resolution),'bicubic')

    def forward(self, text):
        with torch.no_grad():
            if isinstance(text, str):
                clip.tokenize(text) 
                clip_embedding = self.clip.encode_text(text)
            else:        
                image = self.preprocess(text)
                clip_embedding = self.clip.encode_image(image.to(device='cuda:0')) #

        x = self.encoder(clip_embedding.to(torch.float32))
        return x

# segmentation map is resized to match the resolution of the
# corresponding feature map using nearest-neighbor downsampling
class SegmentationEncoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 c = config_encoder_init_channel,
                 stage : int = config_stage, # Maximum # channels for Latent // Base # channels for Latent
                 segmap_shape=config_input_shape,
                  ):
        super(SegmentationEncoder, self).__init__()

        self.ms_out_dims = []
        sw, sh = segmap_shape
        downsamples,embeds, convs = [],[],[]
        for i in range(stage):
            sh, sw = sh//2 , sw//2
            if i != (stage-1):            
                downsamples.append(DownSample(size=(sh,sw)))

            fused = True if  sh >= 128 else False


            embeds.append(ConvBlock(in_channels,c,3,1,downsample=False, fused=fused))
            convs.append(ConvBlock(c, 2*c, 3, 1, downsample=True, fused=fused))

            c = 2*c
            self.ms_out_dims.append(c)
        self.downsamples = nn.ModuleList(downsamples)
        self.embeds = nn.ModuleList(embeds)
        self.convs = nn.ModuleList(convs)

        self.out_dim = c

    def forward(self,seg):
        outputs = [seg] 
        for i, (conv, embed) in enumerate(zip(self.convs, self.embeds)):
            if i == 0 :
                x = conv(embed(seg)) 
                x_d = self.downsamples[i](seg)

            elif i>0:
                x = conv(embed(x_d)+x)
                
                if i != (len(self.downsamples)):
                    x_d= self.downsamples[i](x_d)

            outputs.append(x)

        return outputs 

class SketchEncoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 c = config_encoder_init_channel,
                 stage : int = config_stage, # Maximum # channels for Latent // Base # channels for Latent
                 segmap_shape=config_input_shape, 
                  ):
        super(SketchEncoder, self).__init__()
        self.ms_out_dims = []
        
        sw, sh = segmap_shape
        downsamples,embeds, convs = [],[],[]
        for i in range(stage):
            sh, sw = sh//2 , sw//2
            if i != (stage-1):            
                downsamples.append(DownSample(size=(sh,sw)))

            fused = True if  sh >= 128 else False

            embeds.append(ConvBlock(in_channels,c,3,1,downsample=False, fused=fused))
            convs.append(ConvBlock(c, 2*c, 3, 1, downsample=True, fused=fused))
            c = 2*c
            self.ms_out_dims.append(c)


        self.downsamples = nn.ModuleList(downsamples)
        self.embeds = nn.ModuleList(embeds)
        self.convs = nn.ModuleList(convs)
        self.out_dim = c


    def forward(self,sketch):
        outputs = [sketch]
        for i, (conv, embed) in enumerate(zip(self.convs, self.embeds)):
            if i == 0 :
                x = conv(embed(sketch))
                x_d = self.downsamples[i](sketch)

            elif i>0:
                x = conv(embed(x_d)+x)
                if i != (len(self.downsamples)):
                    x_d= self.downsamples[i](x_d)

            outputs.append(x)
        return outputs




class StyleResblock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 learned_shortcut = True,
                 fused = True,
                  ):
        super(StyleResblock, self).__init__()

        middle_channels = min(in_channels,out_channels)

        self.learned_shortcut = learned_shortcut

        self.norm_0 = nn.InstanceNorm2d(in_channels, affine=False)
        self.norm_1 = nn.InstanceNorm2d(middle_channels, affine=False)

        self.conv_0 = ConvBlock(in_channels, middle_channels, 3, 1, downsample=False, fused=fused) 
        self.conv_1 = ConvBlock(middle_channels, out_channels, 3, 1, downsample=True, fused=fused) # RESNET DOWNSAMPLING IS CON V1X1 AND NORM = IDENTITY

        if self.learned_shortcut:
            self.norm_s = nn.InstanceNorm2d(in_channels, affine=False)
            self.conv_s = ConvBlock(in_channels, out_channels, 3, 1, downsample=True, fused=fused) 

    def forward(self, style):
        x_s = self.shortcut(style)

        x = self.conv_0(self.norm_0(style))
        x = self.conv_1(self.norm_1(x))

        out = x_s + x

        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x))
        else:
            x_s = x
        return x_s


class StyleEncoder(nn.Module):
    def __init__(self,
                 out_channels=512,
                 in_channels =3,
                 stage=config_stage,
                 middle_channels =None,
                  ):
        super(StyleEncoder, self).__init__()
        modules = []
        if middle_channels is None:
            middle_channels = [out_channels//(2**(stage-1-i)) for i in range(stage)]
            
        for middle_channel in middle_channels:
            modules.append(StyleResblock(in_channels,
                 middle_channel))
                
            in_channels = middle_channel

        self.resblocks = nn.ModuleList(modules)

        self.out_dim = sum(middle_channels)
                  
    def forward(self, x):
        mean, log_var = [], []
        for resblock in self.resblocks:
            x = resblock(x)
            x_mean, x_log_var = self.calc_mean_var(x) # 64, 64 / 256, 256 / out_channels, out_channels
            mean.append(x_mean)
            log_var.append(x_log_var)
        
        mu = torch.cat(mean,dim=1)
        logvar = torch.cat(log_var,dim=1)

        return x, [mu, logvar]


    def calc_mean_var(self,feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        feat_size = feat.size()
        assert (len(feat_size) == 4)
        N, C = feat_size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_log_var = torch.log(feat_var).view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_log_var