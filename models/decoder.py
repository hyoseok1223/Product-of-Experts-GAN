
from models.encoder import *
import torch.nn as nn
from models.base_module import PoE, EqualLinear, PixelNorm
import functorch
import torch.nn.functional as F

def logvar_tanh(logvar, theta):
    limit_log_var = theta * torch.tanh((logvar/theta))
    return limit_log_var

class LocalPoeNet(nn.Module):
    def __init__(self, pre_out_channel, spatial_channel
                  ):
        super(LocalPoeNet, self).__init__()
        self.local_poe = PoE()
        self.pre_cnn = CNNHead(pre_out_channel,spatial_channel)
        self.spatial_1_cnn = CNNHead(pre_out_channel+spatial_channel,spatial_channel)
        self.spatial_2_cnn = CNNHead(pre_out_channel+spatial_channel,spatial_channel)

        # if you want to use more spatial modality, add that modality cnn head


    def forward(self, up_pre_out, spatial_feats=None):#seg_feat, sketch_feat

        pre_stats = self.pre_cnn(up_pre_out)
        pre_stats[1] = logvar_tanh(pre_stats[1],theta=1)

        if spatial_feats == None:
            z_k = self.reparameterize(pre_stats[0], pre_stats[1])
        
            return z_k, [pre_stats,pre_stats] 

        spatial_stats = []


        for i, spatial_feat in enumerate(spatial_feats):

            spt_stat = getattr(self, f'spatial_{i+1}_cnn')(torch.cat((up_pre_out,spatial_feat),dim=1))
            spt_stat[1] = logvar_tanh(spt_stat[1],theta=10)
            spatial_stats.append(spt_stat)

        modality_embed_list = [pre_stats, *spatial_stats]
        mu_list,logvar_list = list(zip(*modality_embed_list))

        # apply tanh to poe layer
        mu,logvar = self.local_poe(list(mu_list),list(logvar_list))

        z_k = self.reparameterize(mu, logvar)
        
        return z_k, [pre_stats,[mu,logvar]] # p' , p(z|y) 

    def reparameterize(self, mu, logvar): 
        
        std = torch.exp(logvar).square()
        eps = torch.randn_like(std)

        return eps * std + mu

class LGAdaIN(nn.Module):
    def __init__(self, in_channels, z_channels, w_channels): 
        super(LGAdaIN, self).__init__()
        
        self.param_free_norm = nn.InstanceNorm2d(in_channels, affine=False)
    
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        hidden_dim = 128

        self.conv_shared = ConvBlock(z_channels, hidden_dim, 1, 0)
        self.conv_gamma = EqualConv2d(hidden_dim, in_channels, 1, 1)
        self.conv_beta = EqualConv2d(hidden_dim, in_channels, 1, 1)
        self.conv_gamma.conv.bias.data = torch.ones_like(self.conv_gamma.conv.bias.data)
        self.conv_beta.conv.bias.data  = torch.zeros_like(self.conv_beta.conv.bias.data)
        # pw = ks // 2
        # self.conv_gamma = nn.Conv2d(nhidden, in_channels, kernel_size=ks, padding=pw)
        # self.conv_beta = nn.Conv2d(nhidden, in_channels, kernel_size=ks, padding=pw)

        self.mlp = EqualLinear(w_channels, in_channels * 2)
        self.mlp.linear.bias.data[:in_channels] = 1
        self.mlp.linear.bias.data[in_channels:] = 0

    def forward(self, h_k, z_k, w):

        norm = self.param_free_norm(h_k) 

        local_actv = self.conv_shared(z_k)
        l_gamma = self.conv_gamma(local_actv)
        l_beta = self.conv_beta(local_actv)
        out = norm* l_gamma + l_beta

        global_feat = self.mlp(w).unsqueeze(2).unsqueeze(3)
        g_gamma, g_beta = global_feat.chunk(2, 1)

        out = g_gamma*out + g_beta
        
        return out

class G_ResBlock(nn.Module):
    def __init__(self, pre_out_channel, spatial_channel, w_channel, fused=False, initial=False):
        super(G_ResBlock, self).__init__()

        if initial:
            self.upsample = nn.Identity()
            upsample=False
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upsample= True

        out_channel = pre_out_channel//2 # 1024 -> 512

        self.conv1 = ConvBlock(pre_out_channel,out_channel,3,1,upsample=upsample,fused=fused)
        self.conv2_1 = ConvBlock(pre_out_channel,out_channel,3,1,upsample=upsample,fused=fused)
        self.conv2_2 = ConvBlock(out_channel,out_channel,3,1,upsample=False,fused=fused)

        self.lg_adain = LGAdaIN(out_channel, spatial_channel, w_channel) 
        self.local_poe = LocalPoeNet(pre_out_channel, spatial_channel) 

    def forward(self, pre_output, w, spatial_feats ):

        # 0. upsample
        up_pre_output = self.upsample(pre_output)
        
        # 1. conv ( to residual)
        out_1 = self.conv1(pre_output)

        # 2. conv ( to lgadain)
        h_k = self.conv2_1(pre_output)

        # 3. local poe spatial_feats
        z_k, kl_input = self.local_poe(up_pre_output,spatial_feats)

        h_k = self.lg_adain(h_k,z_k,w)
        h_k = self.conv2_2(h_k)
        out_2 = self.lg_adain(h_k,z_k,w)

        out = out_1+out_2

        return out, kl_input


class GlobalPoeNet(nn.Module):
    def __init__(self, n_mlp_mapping=2, code_dim = 512
                  ):
        super(GlobalPoeNet, self).__init__()

        self.global_poe = PoE()
        self.code_dim = code_dim
        
        mapping_network = [PixelNorm()]
        
        for i in range(n_mlp_mapping):
            mapping_network.append(EqualLinear(code_dim, code_dim))
            mapping_network.append(nn.LeakyReLU(0.2))

        self.mapping_network = nn.Sequential(*mapping_network)
    
    def forward(self, modality_embed_list ):

        mu_list,logvar_list = list(zip(*modality_embed_list))
        
        # apply tanh
        logvar_list = list(logvar_list)
        for i, logvar in enumerate(logvar_list):
            if i == 0:
                logvar_list[i] = logvar_tanh(logvar,theta=1)
            else:
                logvar_list[i] = logvar_tanh(logvar,theta=10)

        
        mu,logvar = self.global_poe(list(mu_list),logvar_list)

        z_0 = self.reparameterize(mu, logvar)

        w = self.mapping_network(z_0)
        
        return w

    def reparameterize(self, mu, logvar): 
        
        std = torch.exp(logvar).square()
        eps = torch.randn_like(std)

        return eps * std + mu

class CNNHead(nn.Module):
    def __init__(self,in_channels,out_channels): 
        super(CNNHead,self).__init__()

        self.out_channels = out_channels

        hideen_dim = in_channels//4

        kernel_size_list = [1,3,3]

        cnn_head = []
        for kernel_size in kernel_size_list:
            if kernel_size == 1:
                padding = 0
            elif kernel_size == 3:
                padding = 1
            cnn_head.append(ConvBlock(in_channels,hideen_dim,kernel_size,padding=padding))
            in_channels = hideen_dim


        cnn_head.append(ConvBlock(hideen_dim, out_channels*2, 1, 0))
        self.cnn_head = nn.Sequential(*cnn_head)


    def forward(self,x):
        x = x.squeeze()
        x = self.cnn_head(x)
        mu = x[:, :self.out_channels,:,:]
        logvar = x[:, self.out_channels:,:,:] 

        return [mu, logvar]

class MlpHead(nn.Module):
    def __init__(self,in_channels,out_channels=512,n_mlp=4,seperate=False, spatial=False): # out_channels mean code dim
        super(MlpHead,self).__init__()

        self.seperate = seperate
        self.spatial = spatial
        self.out_channels = out_channels

        hidden_dim = in_channels//4

        mlp_head = []
        for _ in range(n_mlp-1):
            mlp_head.append(EqualLinear(in_channels, hidden_dim))
            mlp_head.append(nn.LeakyReLU(0.2))
            in_channels = hidden_dim
        if self.seperate:
            self.fc_mu = EqualLinear(hidden_dim, out_channels)
            self.fc_var = EqualLinear(hidden_dim, out_channels)
        else:
            mlp_head.append(EqualLinear(hidden_dim, out_channels*2))
        self.mlp_head = nn.Sequential(*mlp_head)

        

        self.adapool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self,x):
        if self.spatial:
            x = self.adapool(x)


        if self.seperate:
            mu = self.fc_mu(self.mlp_head(x[0].squeeze())) 
            logvar = self.fc_var(self.mlp_head(x[1].squeeze()))
        else:
            x = x.view(x.size(0),-1)
            x = self.mlp_head(x)
            mu = x[:, :self.out_channels]
            logvar = x[:, self.out_channels:] 
            

        return [mu, logvar]


config_input_shape = (256,256) # (1024,1024)
config_stage = 3 # 5
config_spatial_channel = 64 # 32
Maximum_channels_for_Dec = 512 
config_encoder_init_channel 

class Decoder(nn.Module):
    def __init__(self, batch_size,stage=config_stage,pre_out_channel=Maximum_channels_for_Dec,spatial_channel=config_spatial_channel, latent_dim = 512,config_input_shape = config_input_shape[0]# pre_out 아니면 512
                  ):
        super(Decoder, self).__init__()

        self.batch_size = batch_size

        self.latent_dim = latent_dim

        self.text_encoder = TextEncoder()
        self.text_mlp_head = MlpHead(getattr(self.text_encoder,'out_dim'))

        self.seg_encoder = SegmentationEncoder()
        self.seg_mlp_head = MlpHead(getattr(self.seg_encoder,'out_dim'),spatial=True)
        
        self.sketch_encoder = SketchEncoder()
        self.sketch_mlp_head = MlpHead(getattr(self.sketch_encoder,'out_dim'),spatial=True)

        self.style_encoder = StyleEncoder()
        self.style_mlp_head = MlpHead(getattr(self.style_encoder,'out_dim'),seperate=True)

        # define new encoder
        # z_channel

        self.global_poe = GlobalPoeNet()
        prior_mu = torch.zeros(self.latent_dim)
        prior_logvar = torch.log(torch.ones(self.latent_dim))# mu, std
        self.register_buffer('prior_mu',prior_mu)
        self.register_buffer('prior_logvar',prior_logvar)


        self.constant_input = ConstantInput(pre_out_channel,(config_input_shape//(2**stage))) 

        decoder_network = [G_ResBlock(pre_out_channel, spatial_channel, latent_dim, fused=False, initial=True)]

        # for _, channel in zip(range(stage), reversed(getattr(self.seg_encoder, 'ms_out_dims'))):
        for i in range(stage): 
            pre_out_channel = pre_out_channel//2
            spatial_channel = spatial_channel//2
            if i == (stage-1):
                spatial_channel = 1
            decoder_network.append(G_ResBlock(pre_out_channel, spatial_channel, latent_dim, fused=False, initial=False))
        self.decoder = nn.ModuleList(decoder_network)
        self.stage = stage

        self.out_layer = EqualConv2d(pre_out_channel//2,3,1,padding = 0)
        
    # (missing modality embeddings are set to zero)
    def forward(self, modality_dict): 
        N = self.batch_size# list(modality_dict.values())[0].shape[0]
    
        modality_stats = []
        spatial_feat = []

        # modality condition
        if 'text' in  modality_dict.keys():
            text = self.text_encoder(modality_dict['text'])
            text_stats= self.text_mlp_head(text)
            modality_stats.append(text_stats)
        if 'seg_maps' in  modality_dict.keys():
            seg = self.seg_encoder(modality_dict['seg_maps'])
            seg_stats = self.seg_mlp_head(seg[-1])
            modality_stats.append(seg_stats)
            spatial_feat.append(seg)
        if 'sketch_maps' in  modality_dict.keys():
            sketch= self.sketch_encoder(modality_dict['sketch_maps']) # outputs[-1]
            sketch_stats = self.sketch_mlp_head(sketch[-1])
            modality_stats.append(sketch_stats)
            spatial_feat.append(sketch)
        if 'style' in  modality_dict.keys():
            _, style = self.style_encoder(modality_dict['style']) # mu, std
            style_stats = self.style_mlp_head(style)
            modality_stats.append(style_stats)

        # global poenet
        modality_embed_list = [[self.prior_mu.expand(N,self.latent_dim),self.prior_logvar.expand(N,self.latent_dim)],*modality_stats]
        w = self.global_poe(modality_embed_list)

        pre_output = self.constant_input(N)

        kl_inputs=[]
        # contrastive_input = [img_output,text_output] if 'text_output' in locals().keys() else None
        
        for i, resblock in enumerate(self.decoder): # 1024 부터 들어오니까, seg 기준 뒤에서부터 들어와야함.
            spatial_idx = -i-1
            pre_output, kl_input = resblock(pre_output, w, [feat[spatial_idx] for feat in spatial_feat] )
            if len(spatial_feat)>0:
                kl_inputs.append(kl_input)
            else:
                kl_inputs = None

        out = self.out_layer(pre_output)

        return out, kl_inputs # fake image, multi scale mu, logvar 
        
class MPD(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.linear = nn.utils.spectral_norm(EqualConv2d(in_channels, 1, 1, padding=0).conv ,'weight_orig')
        self.linear = EqualConv2d(in_channels, 1, kernel_size=1, padding=0)

    def forward(self,x,y:List = None): 
        out = self.linear(x) # [N,1,H,W]
    

        if y is not None:
            for y_k in y:
                # y_k = torch.sum(y_k*x, dim=1, keepdim=True) # [N,C,H,W] -> [N,1,H,W]

                y_k = torch.mean(y_k*x, dim=1, keepdim=True)

                out += y_k    
        return out # [N,1,H,W] 

class Resblock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 learned_shortcut = True,
                 fused = True,
                  ):
        super(Resblock, self).__init__()

        middle_channels = min(in_channels,out_channels)

        self.learned_shortcut = learned_shortcut

        self.norm_0 = nn.InstanceNorm2d(in_channels, affine=False)
        self.norm_1 = nn.InstanceNorm2d(middle_channels, affine=False)

        self.conv0 = ConvBlock(in_channels, middle_channels, 3, 1, downsample=False, fused=fused) 
        self.conv1 = ConvBlock(middle_channels, out_channels, 3, 1, downsample=True, fused=fused) # RESNET DOWNSAMPLING IS CON V1X1 AND NORM = IDENTITY

        if self.learned_shortcut:
            self.norm_s = nn.InstanceNorm2d(in_channels, affine=False)
            self.convs = ConvBlock(in_channels, out_channels, 3, 1, downsample=True, fused=fused) 

    def forward(self, x):
        x_s = self.shortcut(x)

        x = self.conv0(self.norm_0(x))
        x = self.conv1(self.norm_1(x))

        out = x_s + x

        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.convs(self.norm_s(x))
        else:
            x_s = x
        return x_s



class ImageEncoder(nn.Module):
    def __init__(self,
                 in_channels =3,
                 middle_channel = config_spatial_channel, 
                 stage = config_stage,
                 image_shape = config_input_shape
                  ):
        super(ImageEncoder, self).__init__()
        self.ms_out_dims = []

        modules = []
                 
        sw, sh = image_shape
        for i in range(stage):
            sh, sw = sh//2 , sw//2    
            fused = True if  sh >= 128 else False
            modules.append(Resblock(in_channels,
                 middle_channel, fused = fused))
            in_channels = middle_channel

            self.ms_out_dims.append(middle_channel)

            if i != stage-1:
                middle_channel *= 2


        self.resblocks = nn.ModuleList(modules)
        self.out_dim = middle_channel
                  
    def forward(self, x):
        outputs = []
        for resblock in self.resblocks:

            x = resblock(x)
            outputs.append(x)


        return outputs 


class D_get_logits(nn.Module):
    def __init__(self, linear): 
        super().__init__()

        self.linear = linear

    def forward(self,img_embed, text_embed):
        # img_embed [N, C< H, W]
        # text_embed [N, C]
        img_embed = F.adaptive_avg_pool2d(img_embed, (1,1))# .squeeze(-1).squeeze(-1) -> view
        img_embed = img_embed.view(img_embed.size(0),-1)
        img_embed = self.linear(img_embed)

        return img_embed, text_embed


config_input_shape = (256,256) # (1024,1024)
config_stage = 3 # 5
config_spatial_channel = 64 # 32
config_init_channel = 64

class Discriminator(nn.Module):
    def __init__(self,image_channel=config_spatial_channel,input_shape=config_input_shape[0],stage=config_stage): 
        super().__init__()


        self.input_shape = input_shape

        self.text_encoder = TextEncoder()
        self.seg_encoder = SegmentationEncoder()  
        self.sketch_encoder = SketchEncoder()
        self.style_encoder = StyleEncoder()

        self.image_encoder = ImageEncoder()

        d_channel_list = [image_channel*(2**i) for i in range(stage)] # 32, 64, 128
        self.stage = stage

        mpd_list, text_list, style_list, seg_list, skt_list = [],[],[],[],[]
        
        # multi scale 
        for d_channel, dim in zip(d_channel_list, getattr(self.seg_encoder, 'ms_out_dims')):
            mpd_list.append(MPD(d_channel))
            t,s = self.vector_y_heads(d_channel)
            text_list.append(t)
            style_list.append(s)

            seg,skt = self.spatial_y_heads(dim,d_channel)
            seg_list.append(seg)
            skt_list.append(skt)

        self.mpd_list = nn.ModuleList(mpd_list)
        self.text_list = nn.ModuleList(text_list)
        self.style_list = nn.ModuleList(style_list)
        self.seg_list = nn.ModuleList(seg_list)
        self.skt_list = nn.ModuleList(skt_list)

        contrastive_linear = EqualLinear(getattr(self.image_encoder,'out_dim'), getattr(self.text_encoder,'out_dim'))
        self.img_cond_dnet = D_get_logits(contrastive_linear)
            
    def vector_y_heads(self,d_channel):
        D_text = EqualLinear(getattr(self.text_encoder,'out_dim'), d_channel)
        D_style = EqualLinear(getattr(self.style_encoder,'out_dim')*2, d_channel) # concat logvar, mean

        return  D_text, D_style

    def spatial_y_heads(self,dim, d_channel):
        D_seg = nn.Sequential(ConvBlock(dim, d_channel, 1, 0, downsample=False)
                                        # ,nn.AdaptiveAvgPool2d((1, 1)),nn.Flatten()
                                        )
        D_sketch = nn.Sequential(ConvBlock(dim, d_channel, 1, 0, downsample=False)
                                        # ,nn.AdaptiveAvgPool2d((1, 1)),nn.Flatten()
                                        )

        return D_seg, D_sketch

    def forward(self,x,modality_dict ): # x, y1,y2, y3, y4 #  text, seg, sketch, style
        logits = []

        img_outputs = self.image_encoder(x)

        if 'text' in  modality_dict.keys():
            text_vector = self.text_encoder(modality_dict['text']) # replicate -> repeat
        if 'seg_maps' in  modality_dict.keys():
            seg_outputs = self.seg_encoder(modality_dict['seg_maps'])[1:]
        if 'sketch_maps' in  modality_dict.keys():
            skt_outputs = self.sketch_encoder(modality_dict['sketch_maps'])[1:] # delete sketch
        if 'style' in  modality_dict.keys():
            style_vector = torch.cat(self.style_encoder(modality_dict['style'])[1],dim=1).squeeze() # concat log_var vetor and mean vector


        input_shape = self.input_shape
        for i in range(self.stage):
            scale_modality_output = []
            input_shape = input_shape //2
            if 'seg_maps' in  modality_dict.keys():
                seg_output = self.seg_list[i](seg_outputs[i])
                scale_modality_output.append(seg_output)
            if 'sketch_maps' in  modality_dict.keys():
                skt_output = self.skt_list[i](skt_outputs[i])
                scale_modality_output.append(skt_output)
            if 'text' in  modality_dict.keys():
                text_output = self.text_list[i](text_vector).unsqueeze(-1).unsqueeze(-1).repeat(1,1,input_shape,input_shape)
                scale_modality_output.append(text_output)
            if 'style' in  modality_dict.keys():
                style_output = self.style_list[i](style_vector).unsqueeze(-1).unsqueeze(-1).repeat(1,1,input_shape,input_shape)
                scale_modality_output.append(style_output)

            

            img_output = img_outputs[i]
            logit = self.mpd_list[i](img_output,scale_modality_output)
            
            logits.append(logit)

        contrastive_input = [img_output,text_vector] if 'text_output' in locals().keys() else None

        return logits , contrastive_input # Dx, Dy for contrastive loss