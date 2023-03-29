#%%
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import random

class FolderDataset(Dataset):
    r"""This deals with opening, and reading from an Folder dataset.
    Args:
        root (str): Path to the folder.
    """

    def __init__(self, root, data_type_list,transform, text_description = None, modality_dropout_p = 0.5): # text descroption is a list which is same order with images # sort된상태라 지금은 sort따로 필요없을 듯?
        
        self.transform = transform
        self.root = os.path.join(root)
        self.data_type_list = data_type_list
        self.lengths = []    
        self.p = modality_dropout_p
        for data_type in self.data_type_list:
            if data_type == 'style':
                setattr(self, f'{data_type}_filepaths', glob.glob(os.path.join(self.root, 'images','*')))
            
            elif data_type =='text':
                if text_description:
                    self.text_description = text_description
                else:
                    setattr(self, f'{data_type}_filepaths', glob.glob(os.path.join(self.root, 'images','*')))

            
            else:
                setattr(self, f'{data_type}_filepaths', glob.glob(os.path.join(self.root, data_type,'*')))

            self.lengths.append(len(getattr(self,f'{data_type}_filepaths')))


        def check_equal(lengths):
            return len(set(lengths)) <= 1
        assert check_equal(self.lengths), 'the number of data are not equal each data_type'

        print('Folder at %s opened.' % (root))

    def __getitem__(self, idx): 
        data = {}

        for data_type in self.data_type_list:
            data[data_type] = self.getitem_type(idx,data_type)

        return data 

    def getitem_type(self, idx, data_type):

        if data_type =='text' and hasattr(self, 'text_description'):
            return self.text_description[idx] 


        if data_type == 'style':
            tmp = [i for i in range(self.lengths[0])]
            tmp.remove(idx)
            idx = random.choice(tmp)            

        filepaths = getattr(self, f'{data_type}_filepaths')
        ext = filepaths[0].split('.')[-1] # jpg, png, ...

        if 'JPEG' in ext or 'JPG' in ext  \
                or 'jpeg' in ext  or 'jpg' in ext :
            dtype, mode = np.uint8, 3
        else:
            dtype, mode = np.uint8, -1

        filepath = filepaths[idx]
        assert os.path.exists(filepath), '%s does not exist' % (filepath)

        img = Image.open(filepath).convert('RGB') 
        img = np.array(img)   # RGB

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img[:, :, ::-1]
        
        img = self.transform(image=img)['image']

        if data_type == 'seg_maps' or data_type == 'sketch_maps' :
            img = img[0,:,:].unsqueeze(0)
        return img

    def __len__(self):
        return self.lengths[0]




if __name__ == '__main__':
    from torchvision import datasets, transforms, utils
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2


    transforms=A.Compose([
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                # A.Resize(512,512),
                # A.RandomCrop(256,256), # for 256 size 
                ToTensorV2()
            ])

    dataset = FolderDataset('/home/kmuvcl09/h/tmp_images',['images','seg_maps','sketch_maps', 'style'], transform=transforms)
    print(len(dataset))
    

    utils.save_image(
                    dataset[0]['images'],
                    "sample.png",
                    normalize=True,
                    range=(-1, 1),
                )
