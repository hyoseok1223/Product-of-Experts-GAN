# Preprocessing LHQ Dataset to get landscape image, segmentation map and edge map

This directory implementation is mostly relied on [kazuto1011's deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) and [sniklaus's pytorch-hed](https://github.com/sniklaus/pytorch-hed) 

- Using DeepLAB for segmentation map
- using hed for edgemap

## Setting for Training PoE-GAN
- Preprocess images to segmentation maps and sketch maps
- Using DeepLAB which is trained COCO-Stuff
```sh
python inference.py single\     
    --config-path configs/lhq.yaml\     
    --model-path /data/models/coco_stuff/deeplabv2_resnet/deeplabv2_resnet101_msc-cocostuff164k-100000.pth
    --crf
```

## Data Format
```
|-- LHQ
    |-- lhq_256_jpg
        |-- images
        |-- seg_maps
        |-- sketch_maps
    |-- lhq_1024
        |-- images
        |-- seg_maps
        |-- sketch_maps
```