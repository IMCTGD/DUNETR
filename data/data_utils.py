# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from monai import data, transforms
from monai.data import load_decathlon_datalist
from monai.transforms import LoadImage, LoadImageD
import nibabel as nib
import matplotlib.pyplot as plt
import glob
from utils.utils import reshape_img
import torchvision.transforms as tf
from skimage.filters import ridges

'''
        ：该模块是制作地址字典
        :train_path: 数据的地址
        :ID_list: 选择是训练还是测试
        :label_dir: 标签地址
'''
class get_loader(Dataset):
    def __init__(self, data_dir, label_dir, ID_list, img_size, args, is_train=True):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.ID_list = ID_list
        self.args = args

        self.transform = tf.Compose([
            tf.ToTensor(),
        ])
        self.output_size = img_size

        if is_train :
            # print('-'*20, '使用： 训练数据集 ','-'*20)
            self.image_transform = transforms.Compose(
                [
                    transforms.AddChanneld(keys=["image", "label"]),  ## 增加色彩通道，如果有的话就不增肌了
                    # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),  ## 重定向轴方向
                    # transforms.Spacingd(
                    #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                    # ),    ## 图像重采样，
                    transforms.ScaleIntensityRanged(
                        keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                    ),  ## 图像值强度变换,CT和MRI的值都是从-1000—+3000多的不等，通常需要进行归一化,选定范围区间和归一后的范围
                    # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),  ## 裁剪前景

                    # transforms.RandCropByPosNegLabeld(
                    #     keys=["image", "label"],
                    #     label_key="label",
                    #     spatial_size=(args.roi_x, args.roi_y, args.roi_z),                      ## ROI的大小
                    #     pos=1,                                                                  ## 阳性值的概率
                    #     neg=1,                                                                  ## 阴性值的概率为
                    #     num_samples=4,                                                          ## 返回子图
                    #     image_key="image",
                    #     image_threshold=0,
                    # ),   ## 按阴性阳性比裁剪

                    # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),    ## 随机翻转
                    # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                    # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                    # transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
                    # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                    # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                    # transforms.CenterSpatialCropd(keys=['image', 'label'], roi_size=(256,256,128)), ## 中心裁剪
                    # transforms.ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            # print('-'*20, '使用： 测试数据集 ','-'*20)
            self.image_transform = transforms.Compose(
            [
                transforms.AddChanneld(keys=["image", "label"]),
                # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                # transforms.Spacingd(
                #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                # ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # transforms.ToTensord(keys=["image", "label"]),
            ]
        )

    def __len__(self):
        return len(self.ID_list)

    def __getitem__(self, index):
        image_index = self.ID_list[index]

        image_path = os.path.join(self.data_dir, image_index, 'img.nii.gz')
        # image_path = os.path.join(self.data_dir, image_index, 'frangi.nii.gz')
        label_path = os.path.join(self.label_dir, image_index, 'label.nii.gz')

        # image_path = 'F:/raw_dataset/train/data/volume-0.nii'
        # label_path = 'F:/raw_dataset/train/label/segmentation-0.nii'

        dict_loader = LoadImageD(keys=("image", "label"), image_only=False)
        data_dict = dict_loader({"image": image_path,
                                 "label": label_path})      ## 原始图像

        tf_data_dict = self.image_transform(data_dict)      ## 做了变换

        img_size = np.array(data_dict['image'].shape)       ## 原始图像的尺寸大小

        image, labels = data_dict["image"], data_dict["label"]

        imagecrop, labelcrop = tf_data_dict["image"], tf_data_dict["label"]
        # print(imagecrop.shape)
        # print('*!!'*20)
        # print(qw)

        img = reshape_img(imagecrop.squeeze(0), self.output_size)  ## 改变图像的大小与格
        label = reshape_img(labelcrop.squeeze(0), self.output_size)

        plt.figure("visualize", (8, 8))
        # for i in range(int(image.shape[2])):
        #     plt.imshow(imagecrop[0,:, :, i], cmap="gray")
        #     plt.savefig(f'./result/Direct_seg/UNet3D/{i}_1.jpg')
        # print(qwe)

        a = 155
        plt.subplot(1, 2, 1)
        plt.title("Original image")
        plt.imshow(image[:, :, a], cmap="gray")
        plt.imshow(labels[:, :, a]+155,alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.title("Normal image")
        plt.imshow(imagecrop[0,:, :, a], cmap="gray")
        plt.imshow(labelcrop[0,:, :, a]+155,alpha=0.3)

        plt.show()
        plt.savefig(f'./result/Direct_seg/UNet3D/0000_1.jpg')
        #
        print(qwe)


        # a = 45
        # print(image.shape)
        # print(labels.shape)
        # print(imagecrop.shape)
        # print(labelcrop.shape)
        # plt.figure("visualize", (8, 8))
        # plt.subplot(2, 2, 1)
        # plt.title("image")
        # plt.imshow(image[:, :, a], cmap="gray")
        # plt.imshow(labels[:, :, a]+255,alpha=0.2)
        # plt.subplot(2, 2, 2)
        # plt.title("label")
        # plt.imshow(labels[:, :, a])
        # plt.subplot(2, 2, 3)
        # plt.title("dealed image")
        # plt.imshow(image[:, :, a], cmap="gray")
        # plt.subplot(2, 2, 4)
        # plt.title("dealed label")
        # plt.imshow(label[:, :, a])
        # plt.show()
        # print(qw)

        img = self.transform(img).unsqueeze(0)
        label = self.transform(label).unsqueeze(0)

        sample = {'image': img, 'label': label, 'affine': tf_data_dict['image_meta_dict']['affine'], 'image_index': image_index, 'image_size': img_size}

        # print(image_index)
        # print(imagecrop.squeeze(0).numpy().shape)
        # print(imagecrop.squeeze(0).numpy().dtype)
        # nib.Nifti1Image(imagecrop.squeeze(0).numpy(), tf_data_dict['image_meta_dict']['affine'])\
        #     .to_filename('./pre_label.nii.gz')
        ## 显示2*2的图像
        # print(img.shape)
        # print(data_dict['image'].shape)
        # print("*@#$%^&*" * 20)
        # print(qw)

        return sample

if __name__ == '__main__':
    print('star-------')


