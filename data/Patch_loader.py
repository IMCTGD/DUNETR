from torch.utils.data import DataLoader, Dataset
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom
import os
import pandas as pd
from monai.transforms import RandFlip, RandRotate
from utils.utils import reshape_img,normalize

# from sklearn.model_selection import KFold
"""
该程序为读取数据的类
"""
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class CoronaryImagePatch(Dataset):

    def __init__(self, data_dir, label_dir,enhance_path=None,transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.label_dir = label_dir
        self.data_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_index = self.data_list[index]

        image_path = os.path.join(self.data_dir, image_index)
        label_path = os.path.join(self.label_dir, image_index)

        ID = image_index.split('.')[0]

        img_nii = nib.load(image_path)
        img = img_nii.get_data()
        label = nib.load(label_path).get_data()

        img = np.expand_dims(img, 0)
        label = np.expand_dims(label, 0)
        img = normalize(img)
        concatenate = np.concatenate([img, label], axis=0)

        if self.transform != None:
            concatenate = self.transform(concatenate)
        img = concatenate[0:1, :, :, :]
        label = concatenate[1:, :, :, :]
        sample = {'img': img, 'label': label, 'affine': img_nii.affine, 'image_index': ID}
        return sample


def transform(img):
    tran1 = RandFlip(prob=0.5)
    img = tran1(img)
    tran2 = RandRotate(prob=0.5)
    img = tran2(img)
    return img


class CoronaryImageEnhance(Dataset):
    def __init__(self, data_dir, enhance_dir, label_dir, transform=transform):
        self.data_dir = data_dir
        self.enhance = enhance_dir
        self.transform = transform
        self.label_dir = label_dir
        self.data_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_index = self.data_list[index]

        image_path = os.path.join(self.data_dir, image_index)
        label_path = os.path.join(self.label_dir, image_index)
        enhance_path = os.path.join(self.enhance, image_index)

        ID = image_index.split('.')[0]

        img_nii = nib.load(image_path)

        img = img_nii.get_data()
        enhance = nib.load(enhance_path).get_data()
        label = nib.load(label_path).get_data()

        img = np.expand_dims(img, 0)
        enhance = np.expand_dims(enhance, 0)
        label = np.expand_dims(label, 0)
        concatenate = np.concatenate([img, enhance, label], axis=0)
        if self.transform != None:
            concatenate = self.transform(concatenate)
        img = concatenate[0:2, :, :, :]
        label = concatenate[2:, :, :, :]
        sample = {'img': img, 'label': label, 'affine': img_nii.affine, 'image_index': ID}
        return sample


class CoronaryImageSigle(Dataset):
    def __init__(self, data_dir, enhance_dir, label_dir, output_size, transform=transform):
        self.data_dir = data_dir
        self.enhance = enhance_dir
        self.transform = transform
        self.label_dir = label_dir
        self.data_list = os.listdir(data_dir)
        self.output_size = output_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        image_index = self.data_list[index]
        image_path = os.path.join(self.data_dir, image_index)
        label_path = os.path.join(self.label_dir, image_index)

        ID = image_index.split('.')[0]

        img_nii = nib.load(image_path)

        img = img_nii.get_data()
        label = nib.load(label_path).get_data()

        img = reshape_img(img, self.output_size)
        label = reshape_img(label, self.output_size)
        label[label > 0] = 1

        img = normalize(img)

        img = np.expand_dims(img, 0)
        label = np.expand_dims(label, 0)
        concatenate = np.concatenate([img, label], axis=0)
        if self.transform != None:
            concatenate = self.transform(concatenate)
        img = concatenate[0:1, :, :, :]
        label = concatenate[1:, :, :, :]
        sample = {'img': img, 'label': label, 'affine': img_nii.affine, 'image_index': ID}
        return sample


