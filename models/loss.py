import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(u, v,device, temperature=0.3): 
    # normalized features   
    u = u / u.norm(dim=-1,keepdim=True)
    v = v / v.norm(dim=-1,keepdim=True)

    logits_u =  u @ v.t() / temperature
    logits_v =  v @ u.t() / temperature 
    labels = torch.arange(len(logits_u))
    loss_first_term = F.cross_entropy(logits_u, labels)
    loss_second_term  = F.cross_entropy(logits_v, labels)
    loss = (loss_first_term + loss_second_term) / 2 

    return loss 


# https://discuss.pytorch.org/t/kl-divergence-between-two-multivariate-gaussian/53024/17
def kl_divergence(mu1, sigma_1, mu2, sigma_2):

    mu1, sigma_1, mu2, sigma_2 = F.adaptive_avg_pool2d(mu1, (1,1)), F.adaptive_avg_pool2d(sigma_1, (1,1),),F.adaptive_avg_pool2d(mu2, (1,1)),F.adaptive_avg_pool2d(sigma_2, (1,1))
    mu1, sigma_1, mu2, sigma_2 = mu1.view(mu1.size(0),-1), sigma_1.view(mu1.size(0),-1), mu2.view(mu1.size(0),-1), sigma_2.view(mu1.size(0),-1)

    cov1 = torch.stack([torch.diag(sigma) for sigma in torch.exp(sigma_1)])
    mvn1 = torch.distributions.MultivariateNormal(mu1, cov1)

    cov2 = torch.stack([torch.diag(sigma) for sigma in torch.exp(sigma_2)])
    mvn2 = torch.distributions.MultivariateNormal(mu2, cov2)

    return torch.distributions.kl_divergence(mvn1, mvn2)

def multi_res_kl_div_loss(kl_inputs):
    alpha = kl_balancer_coeff(len(kl_inputs))
    kl_all = []
    for kl_input in kl_inputs:
        kl_input = kl_input[0]+kl_input[1]
        kl = kl_divergence(*kl_input)
        kl_all.append(kl)
    kl_all = torch.cat(kl_all) # torch.cat is fast way list to tensor
    kl_loss,kl_coeffs = kl_balancer(kl_all, alpha=alpha)
    return kl_loss # kl_coeffs is for running average kl rebalancing weights


def kl_balancer_coeff(num_scales, fun='linear'):
    if fun == 'equal':
        coeff = torch.cat([1 for i in range(num_scales)]).cuda()
    elif fun == 'linear':
        coeff = torch.cat([(2 ** i)*torch.ones(1)  for i in range(num_scales)]).cuda()
    elif fun == 'sqrt':
        coeff = torch.cat([np.sqrt(2 ** i) for i in range(num_scales)]).cuda()
    elif fun == 'square':
        coeff = torch.cat([np.square(2 ** i) / num_scales  for i in range(num_scales)]).cuda()
    else:
        raise NotImplementedError
    # convert min to 1.
    coeff /= torch.min(coeff)
    return coeff

def kl_balancer(kl_all, kl_coeff=1.0,alpha=None): 
    kl_coeff = kl_coeff / alpha * kl_all # element wise multiplicatoin
    total_kl = torch.sum(kl_coeff)

    kl_coeff = kl_coeff / total_kl

    assert sum(kl_coeff) == 1

    kl = torch.sum(kl_all * kl_coeff.detach(), dim=1)
    # for reporting
    # kl_coeffs = kl_coeff_i.squeeze(0)
    
    return kl, kl_coeff 
