import numpy as np
import nibabel as nib
import time
import os
import pandas as pd
import multiprocessing
from skimage.morphology import skeletonize


def get_patch(image, label, enhance, patch_size):

    '''
    根据真实标签中心线来选取训练集，训练集有无标签比例为1：1
    :param image: 真实图像
    :param label: 真实标签
    :param flag_label: 先验区域
    :param patch_size: patch的大小
    :return: data_patch_list:图像patch合集 ;label_patch_list,标签patch合集; loc_record patch位置合集
    '''

    flag_label = label
    shape = flag_label.shape
    shape = np.array(shape)
    label_patch_list = []
    data_patch_list = []
    enhance_patch_list = []
    flag_record = skeletonize(flag_label.astype(np.uint8))
    i, j, k = np.where(flag_record == 1)
    loc_record = []

    # 沿着中心线裁剪
    pos = 0
    neg = 0
    for index in range(i.shape[0]):
        if flag_record[i[index], j[index], k[index]] == 1:
            c = np.array([i[index], j[index], k[index]])
            s = c - patch_size // 2
            e = c + patch_size // 2
            z_s = np.zeros_like(s)

            e[s < z_s] = patch_size
            s[s < z_s] = 0

            s[e > shape] = shape[e > shape] - patch_size
            e[e > shape] = shape[e > shape]

            label_patch = label[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
            label_patch_list.append(label_patch)

            data_patch = image[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
            data_patch_list.append(data_patch)

            enhance_patch = enhance[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
            enhance_patch_list.append(enhance_patch)

            loc_record.append(s)
            flag_record[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = 0
            pos = pos + 1
    x = np.arange(patch_size // 2, shape[0], patch_size)
    y = np.arange(patch_size // 2, shape[1], patch_size)
    z = np.arange(patch_size // 2, shape[2], patch_size)
    i, j, k = np.meshgrid(x, y, z)
    i, j, k = i.flatten(), j.flatten(), k.flatten()
    arr = np.arange(i.shape[0])
    np.random.shuffle(arr)
    arr = arr.tolist()

    for index in arr:
        c = np.array([i[index], j[index], k[index]])
        s = c - patch_size // 2
        e = c + patch_size // 2
        z_s = np.zeros_like(s)

        e[s < z_s] = patch_size
        s[s < z_s] = 0

        s[e > shape] = shape[e > shape] - patch_size
        e[e > shape] = shape[e > shape]

        label_patch = label[s[0]:e[0], s[1]:e[1], s[2]:e[2]]

        if label_patch.sum() == 0:
            label_patch_list.append(label_patch)
            data_patch = image[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
            data_patch_list.append(data_patch)

            enhance_patch = enhance[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
            enhance_patch_list.append(enhance_patch)

            loc_record.append(s)
            neg = neg + 1

        if neg >= pos:
            break


    return data_patch_list, label_patch_list, enhance_patch_list, loc_record


def get_patch_valid(image, label, enhance, patch_size):
    '''
    规则裁剪，制作重复区域
    :param image:
    :param label:
    :param enhance:
    :param patch_size:
    :return:
    '''

    shape = label.shape
    shape = np.array(shape)
    label_patch_list = []
    data_patch_list = []
    enhance_patch_list = []

    img_record = label.copy()
    # i,j,k=np.where(label==1)
    x = np.arange(patch_size // 2, shape[0], patch_size // 2)
    y = np.arange(patch_size // 2, shape[1], patch_size //2)
    z = np.arange(patch_size // 2, shape[2], patch_size //2)
    i, j, k = np.meshgrid(x, y, z)
    i, j, k = i.flatten(), j.flatten(), k.flatten()

    loc_record = []

    for index in range(i.shape[0]):
        c = np.array([i[index], j[index], k[index]])
        s = c - patch_size // 2
        e = c + patch_size // 2
        z_s = np.zeros_like(s)

        e[s < z_s] = patch_size
        s[s < z_s] = 0

        s[e > shape] = shape[e > shape] - patch_size
        e[e > shape] = shape[e > shape]

        label_patch = label[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
        label_patch_list.append(label_patch)

        data_patch = image[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
        data_patch_list.append(data_patch)

        enhance_patch_list.append(enhance[s[0]:e[0], s[1]:e[1], s[2]:e[2]])

        loc_record.append(s)

        img_record[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = 0

    return data_patch_list, label_patch_list, enhance_patch_list, loc_record


class Get_patch:
    def __init__(self, data_path, label_path, frangi_path, save_path, patch_size, data_type):
        self.data_path = data_path
        self.label_path = label_path
        self.frangi_path = frangi_path
        self.save_img_path = os.path.join(save_path, 'patch_%d' % patch_size, '%s_img_patch' % data_type)
        self.save_label_path = os.path.join(save_path, 'patch_%d' % patch_size, '%s_label_patch' % data_type)
        self.save_frangi_path = os.path.join(save_path, 'patch_%d' % patch_size, '%s_enhance_patch' % data_type)
        self.save_record_path = os.path.join(save_path, 'patch_%d' % patch_size, 'csv_patch_record')
        self.patch_size = patch_size
        self.data_type = data_type

        os.makedirs(self.save_img_path,exist_ok=True)
        os.makedirs(self.save_label_path,exist_ok=True)
        os.makedirs(self.save_frangi_path,exist_ok=True)
        os.makedirs(self.save_record_path,exist_ok=True)

    def run_main(self, i):
        print(i)
        df = pd.DataFrame(columns=['x', 'y', 'z'])
        d_path = os.path.join(self.data_path, i, 'img.nii.gz')
        l_path = os.path.join(self.label_path, i, 'label.nii.gz')
        e_path = os.path.join(self.frangi_path, i, 'frangi.nii.gz')

        d_nii = nib.load(d_path)
        l_nii = nib.load(l_path)
        e_nii = nib.load(e_path)

        p_label = l_nii.get_fdata()

        p_label[p_label > 0] = 1
        p_label[p_label <= 0] = 0

        img = d_nii.get_fdata()
        enhance = e_nii.get_fdata()

        if self.data_type == 'train':
            g_p = get_patch
        else:
            g_p = get_patch_valid

        img_list, label_list, enhance_list, loc_record = g_p(img, p_label, enhance, self.patch_size)
        ID = i
        affine = l_nii.affine
        for index, p in enumerate(img_list):
            nib.save(nib.Nifti1Image(p, affine), os.path.join(self.save_img_path, ID + '_%d.nii.gz' % index))
            nib.save(nib.Nifti1Image(label_list[index], affine),
                     os.path.join(self.save_label_path, ID + '_%d.nii.gz' % index))
            df.loc[ID + '_%d.nii.gz' % index] = [loc_record[index][0], loc_record[index][1], loc_record[index][2]]
            nib.save(nib.Nifti1Image(enhance_list[index], affine),
                     os.path.join(self.save_frangi_path, ID + '_%d.nii.gz' % index))
            df.to_csv(os.path.join(self.save_record_path, ID + '.csv'))

