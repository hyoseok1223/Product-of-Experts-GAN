
from .encoder import *
import torch.nn as nn
from .base_module import PoE, EqualLinear, PixelNorm

class GlobalPoeNet(nn.Module):
    def __init__(self, n_mlp_mapping=2, n_mlp_head = 4, code_dim = 512
                  ):
        super(GlobalPoeNet, self).__init__().
        self.global_poe = PoE()
        self.code_dim = code_dim
        
        mapping_network = [PixelNorm()]
        for i in range(n_mlp_mapping):
            mapping_network.append(EqualLinear(code_dim, code_dim))
            mapping_network.append(nn.LeakyReLU(0.2))

        self.mapping_network = nn.Sequential(*mapping_network)
    
    def forward(self, modality_embed_list ): 

        mu_list,std_list = list(zip(*modality_embed_list))
        
        mu,std = self.global_poe(list(mu_list),list(std_list))

        z_0 = self.reparameterize(mu, std)

        w = self.mapping_network(z_0)
        
        return w

    def reparameterize(self, mu, std):

        eps = torch.randn_like(std)
        return eps * std + mu
        


