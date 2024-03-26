from torch.utils.data import DataLoader, Dataset
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom
import os
import torch
import pandas as pd
# from monai.transforms import RandFlip, RandRotate
from utils.utils import reshape_img,normalize
from torchvision.transforms import transforms
import random

to_pil_image = transforms.ToPILImage()

tf = transforms.Compose([
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225],
    #                      ),
    transforms.ToTensor(),

])

class RandomCrop_3d:
    def __init__(self, slices):
        self.slices = slices

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def __call__(self, img, mask):

        ss, es = self._get_range(mask.size(0), self.slices)

        tmp_img = torch.zeros((img.size(0), self.slices, img.size(2),img.size(3)))
        tmp_mask = torch.zeros((mask.size(0), self.slices, mask.size(2),mask.size(3)))

        tmp_img[:, :es - ss] = img[:, ss:es]
        tmp_mask[:, :es - ss] = mask[:, ss:es]
        return tmp_img, tmp_mask

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

## 主程序
class CoronaryImage(Dataset):
    def __init__(self, data_dir, label_dir, ID_list, img_size, transform=tf, is_normal=True):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.ID_list = ID_list

        self.transform = transform
        # self.data_list = os.listdir(data_dir)
        self.output_size = img_size
        self.is_normal = is_normal

        self.RandomCrop_3d = RandomCrop_3d(64)
        # self.RandomFlip_LR = RandomFlip_LR(prob=0.5)

    def __len__(self):
        return len(self.ID_list)

    def __getitem__(self, index):
        image_index = self.ID_list[index]

        image_path = os.path.join(self.data_dir, image_index, 'ture_img.nii.gz')
        pre_image_path = os.path.join(self.data_dir, image_index, 'pre_img.nii.gz')
        label_path = os.path.join(self.label_dir, image_index, 'pre_label.nii.gz')

        turel_path = os.path.join(self.label_dir, image_index, 'label_img.nii.gz')

        ID = image_index

        img_nii = nib.load(image_path)
        img = img_nii.get_data()

        pre_img_nii = nib.load(pre_image_path)
        pre_img = pre_img_nii.get_data()

        label_nii = nib.load(label_path)
        label = label_nii.get_data()

        ture_nii = nib.load(turel_path)
        ture = ture_nii.get_data()

        # print('ture:',ture.shape)
        # print('label:',label.shape)
        # print('img:',img.shape)
        #
        # panduan = img-label-ture
        # panduan[panduan>0] = 3
        # # panduan[panduan==0] = 2
        # panduan[panduan<0] = 1
        # print(image_index)
        # pre_label_new = nib.Nifti1Image(panduan, img_nii.affine, img_nii.header)
        # nib.save(pre_label_new, f'./111111.nii.gz')
        #
        # a,b = np.unique(panduan,return_counts=True)
        # print(a,b)
        # print(qwe)

        img_size = np.array(img.shape)

        img = reshape_img(img, self.output_size)        ## 改变图像的大小与格
        pre_img = reshape_img(pre_img, self.output_size)
        label = reshape_img(label, self.output_size)
        ture = reshape_img(ture, self.output_size)

        if self.is_normal == True:
            img = normalize(img)

        img = self.transform(img).unsqueeze(0)
        pre_img = self.transform(pre_img).unsqueeze(0)
        label = self.transform(label).unsqueeze(0)
        ture = self.transform(ture).unsqueeze(0)

        sample = {'image': img, 'pre_img':pre_img, 'label': label, 'ture':ture, 'affine': img_nii.affine, 'image_index': ID, 'image_size': img_size}
        return sample