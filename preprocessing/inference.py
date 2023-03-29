#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019

from __future__ import absolute_import, division, print_function

import click 
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import os.path as osp

from libs.models import *
from libs.utils import DenseCRF

from PIL import Image
from libs.datasets import get_dataset
from tqdm import tqdm

from libs.models import hed
import numpy

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor


def preprocessing(image, device, CONFIG):
    # Resize
    print(image)
    scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    image = cv2.resize(image, (256,256))# fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def inference(model, image, raw_image=None, postprocessor=None):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    #FIXME - Select some classes like sea, sky, ...
    logits = logits[:,[156,105,123,177,149,134,168,153,158,154,147,126,124],:,:]
    probs = F.softmax(logits, dim=1)
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        post_probs = []
        for raw_img, prob in zip(raw_image, probs):
            prob = postprocessor(raw_img.cpu().numpy().transpose(1,2,0), prob)
            post_probs.append(prob)

        probs = np.stack(post_probs)
    labelmap = np.argmax(probs, axis=1) 
    labelmap = scale(labelmap)
    return labelmap


@click.group()
@click.pass_context
def main(ctx):
    """
    Demo with a trained model
    """

    print("Mode:", ctx.invoked_subcommand)


@main.command() 
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
@click.option("--crf", is_flag=True, show_default=True, help="CRF post-processing")
def single(config_path, model_path, cuda, crf):
    """
    Inference from a single image
    """

    # Setup
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=16, # must be set 1 for crf
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
    )

    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    # Inference
    root = CONFIG.DATASET.ROOT
    
    for image_ids, image, _ in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        raw_image = image.to(torch.uint8).to(device)
        image = image.to(device)
        labelmaps = inference(model, image, raw_image, postprocessor)
        sketchs = hed.estimate(image)

        for image_id, labelmap, sketch in zip(image_ids, labelmaps, sketchs):
            seg_map = Image.fromarray(labelmap.astype(np.uint8))
            seg_map.save(osp.join(root,'seg_maps',f"{image_id}.jpg"))

            sketch_map = Image.fromarray((sketch.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8))
            sketch_map.save(osp.join(root,'sketch_maps',f"{image_id}.jpg"))

def scale(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x_scaled = 255*(x-x_min)/(x_max-x_min)
    return x_scaled
if __name__ == "__main__":
    main()
