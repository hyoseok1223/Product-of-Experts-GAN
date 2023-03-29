import sys
sys.settrace
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from models import decoder as model
from models.loss import contrastive_loss, multi_res_kl_div_loss
from dataset.dataset import FolderDataset
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn, autograd, optim

from torch.cuda.amp import autocast, GradScaler

from torch.autograd import grad as torch_grad

import timm 

import wandb
import random

import warnings
warnings.filterwarnings(action='ignore')

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# ema!
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_( par2[k].data,alpha=1 - decay)


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() , fake_loss.mean()

def gradient_penalty(real_img,real_pred):
    # with conv2d_gradfix.no_weight_gradients():
    
    grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True, retain_graph=True, only_inputs=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

class VGG_contrastive(nn.Module):
    def __init__(self): # d_channel은 text vector의 channel size 디폴트 512
        super().__init__()
        vgg19 = timm.create_model('vgg19', pretrained=True).features[:29+1].to(device) # vgg19 relu5_1
        self.vgg19 = vgg19.eval()

    def forward(self,x):
        # img_embed [N, C< H, W]
        # text_embed [N, C]
        x = self.vgg19(x)
        x = F.adaptive_avg_pool2d(x, (1,1))# .squeeze(-1).squeeze(-1) -> view
        x = x.view(x.size(0),-1)

        return x


# https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py
def gradient_penalty(output, images, weight = 10):
    batch_size = images.shape[0]

    gradients = grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device), # output.size()
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def mmpd_g_nonsaturating_loss(fake_pred): # fake perd has [scale1_logits, scale2_logits, ..]
    multi_scale_loss = []
    for scale_fake_pred in fake_pred:
        multi_scale_loss.append(g_nonsaturating_loss(scale_fake_pred))
    loss = torch.mean(torch.stack(multi_scale_loss))

    return loss

def mmpd_d_logistic_loss(real_pred, fake_pred):
    multi_scale_real_loss = []
    multi_scale_fake_loss = []
    # print(fake_pred)
    for scale_real_pred, scale_fake_pred in zip(real_pred,fake_pred):
        real_loss,fake_loss = d_logistic_loss(scale_real_pred,scale_fake_pred)
        multi_scale_real_loss.append(real_loss)
        multi_scale_fake_loss.append(fake_loss)



    return torch.mean(torch.stack(multi_scale_real_loss)) + torch.mean(torch.stack(multi_scale_fake_loss))

# for iter training
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def train(dataloader,generator,g_ema, discriminator,g_optim, d_optim, device, *lambda_list ):
    
    print('start')
    lambda1, lambda2, lambda3 = lambda_list

    max_iter = 800000

    # pbar setting
    loader = sample_data(dataloader)
    pbar = range(max_iter)
    pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=0.01)

    accum_steps = 16

    for idx in pbar:
        idx = idx +start_iter
        if idx > max_iter:
            print('done')
            break
        data = next(loader)

        p = 0.5
        real_img = data['images'].to(device)
        data.pop('images')
        for data_type in list(data.keys()):
            data[data_type] = data[data_type].to(device)
            if random.random() < p :
                data.pop(data_type)

        # lazy regularization
        d_regularize = idx % 16 == 0 # 16

        # discriminator step
        requires_grad(generator, False)
        requires_grad(discriminator, True)


        d_loss = 0
        g_loss = 0

        if True:
            fake_img, _ = generator(data)

            real_pred, d_contrastive_y_input = discriminator(real_img, data)
            fake_pred, _ = discriminator(fake_img, data) 
            # gan loss
            d_logistic_loss = mmpd_d_logistic_loss(real_pred, fake_pred) # non-saturated gan loss

            d_loss += d_logistic_loss

            # contrastive y loss
            d_contrastive_y_loss=0
            #https://github.com/google-research/xmcgan_image_generation
            if d_contrastive_y_input is not None and idx>0:
                d_contrastive_y_loss=contrastive_loss(*discriminator.img_cond_dnet(*d_contrastive_y_input),device=device)
                d_loss += lambda2 * d_contrastive_y_loss
                

        d_loss.backward()    
        nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10.0)  
        if idx % accum_steps == 0:
            d_optim.step()
            d_optim.zero_grad()
            
        if True:
            # lazy regularization
            if d_regularize:
                real_img.requires_grad = True
                real_pred, _ = discriminator(real_img, data)
                real_pred_gp = torch.stack(list(map(lambda tensor : torch.mean(tensor) , real_pred))).sum()
                gp = gradient_penalty(real_pred_gp, real_img) 
                d_loss += lambda3 *gp    
                # 이거 사이즈 안맞는 경우 뭐지?? 설마 특정 scale이 dropout되나..?         
                #        
                d_optim.zero_grad()
                (lambda3 * gp * d_reg_every).backward()

                d_optim.step()
                    
        # generator step
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        if True:

            fake_img, kl_inputs = generator(data)
            fake_pred, g_contrastive_y_input = discriminator(fake_img,data)

            

            # GAN loss
            g_nonsaturating_loss = mmpd_g_nonsaturating_loss(fake_pred)

            g_loss += g_nonsaturating_loss

            g_contrastive_x_loss = 0
            if idx>0:
                vgg19 = VGG_contrastive()

                g_contrastive_x_loss = contrastive_loss(vgg19(real_img),vgg19(fake_img), device=device)

                g_loss +=  lambda1 * g_contrastive_x_loss

            # contrastive y loss
            g_contrastive_y_loss=0
            if g_contrastive_y_input is not None and idx>0:
                g_contrastive_y_loss=contrastive_loss(*discriminator.img_cond_dnet(*g_contrastive_y_input), device=device)
                g_loss += lambda2 * g_contrastive_y_loss

            # kl loss
            if kl_inputs is not None and False:
                g_kl_loss = multi_res_kl_div_loss(kl_inputs)
                g_loss += g_kl_loss 


        g_loss.backward()    
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)  
        if idx % accum_steps == 0:
            g_optim.step()
            g_optim.zero_grad()
            

        accumulate(g_ema, generator)
        
            
        pbar.set_description((
                        f"d: {d_loss:.4f}; g: {g_loss:.4f};"
                        f"d_logistic_loss: {d_logistic_loss:.4f}; d_contrastive_y_loss: {d_contrastive_y_loss:.4f}; gp: {gp:.4f};"
                        f"g_nonsaturating_loss: {g_nonsaturating_loss:.4f}; g_contrastive_y_loss: {g_contrastive_y_loss:.4f}; g_contrastive_x_loss: {g_contrastive_x_loss:.4f};"# g_kl_loss: {g_kl_loss:.4f}; "
                    ))



        if idx % 100 == 0:
            with torch.no_grad():
                g_ema.eval()
                sample, _ = g_ema(data)
                utils.save_image(
                    sample,
                    f"sample/{str(idx).zfill(6)}.png",
                    # nrow=int(args.n_sample ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )

        # weight save
        if idx % 4000 == 0:
            torch.save(
                {
                    "g": generator.state_dict(),
                    "d": discriminator.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    # "args": args,
                },
                f"checkpoint/{str(idx).zfill(6)}.pt",
            )

p = 0.5
def moality_dropout_collate_fn(batch):

    for data_type in list(batch[0].keys()):
        if data_type == 'images':
            continue

        if random.random() < p :
            batch.pop(data_type)
    return batch

if __name__ == '__main__':

    device = "cuda"
    batch_size = 4
    generator = model.Decoder(batch_size
        # args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = model.Discriminator(
        # args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = model.Decoder(batch_size
        # args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)


    d_reg_every = 16
    d_reg_ratio = d_reg_every / (d_reg_every + 1)
    lr =0.004#/8

    g_optim = optim.Adam(
        generator.parameters(),
        lr=lr,
        betas=(0 , 0.99 ),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=lr* d_reg_ratio,
        # betas=(0 , 0.99 ),
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    
    start_iter = 0
    import os 
    ckpt_path = None#'/home/kmuvcl09/h/checkpoint/020000.pt'
    
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        ckpt_name = os.path.basename(ckpt_path)
        start_iter = int(os.path.splitext(ckpt_name)[0])
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])




    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms=A.Compose([
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                # A.Resize(512,512),
                # A.RandomCrop(256,256), # for 256 size 
                ToTensorV2()
            ])
    dataset = FolderDataset('/home/kmuvcl09/h/LHQ/lhq_256_jpg_s',['images','text','seg_maps','sketch_maps', 'style'],transform=transforms)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) # collate_fn = moality_dropout_collate_fn

    # wandb
    # if get_rank() == 0 and wandb is not None and args.wandb:
    #     wandb.init(project="stylegan 2")

    # train
    train(dataloader,generator,g_ema, discriminator,g_optim, d_optim, device,3,0.3,1)