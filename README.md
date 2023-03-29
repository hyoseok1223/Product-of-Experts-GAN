# Product-of-Experts-GAN

This Repository is unoffical implementation of [PoE-GAN : Multimodal Conditional Image Synthesis with Product-of-Experts GANs](https://arxiv.org/abs/2112.05130) in PyTorch
> This implementation is mostly relied on "[rosinality's stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)"

> ![image](https://user-images.githubusercontent.com/90104418/228483420-9a1a0cbf-f0f4-4ed0-88d3-64ca071f50aa.png)


## Notice
I have tried to follow offical paper experiment settings, but some parts are not perfectyl implemented. ( TBD! )
- It works when enough batch size
- Train only non-saturating gan loss first, and from intermediate steps using modality loss ( contrastive loss and KL loss )

## Requirements
```
> CUDA 11.3
> pytorch-hed
> clip-pytorch
```

## Sample
![image](https://user-images.githubusercontent.com/90104418/228483826-a541dc90-e8ab-4ade-adbd-f1591d072756.png)
