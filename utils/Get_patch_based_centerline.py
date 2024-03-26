import numpy as np
import nibabel as nib
import os
import pandas as pd
from skimage.morphology import skeletonize


def get_patch(image, label,flag_label, patch_size):
    '''
    :param image: 真实图像
    :param label: 真实标签
    :param flag_label: 先验区域
    :param patch_size: patch的大小
    :return: data_patch_list:图像patch合集 ;label_patch_list,标签patch合集; loc_record patch位置合集
    '''

    shape = flag_label.shape
    shape = np.array(shape)
    label_patch_list = []
    data_patch_list = []
    flag_record = skeletonize(flag_label.astype(np.uint8))
    i, j, k = np.where(flag_record == 1)
    loc_record = []

    # 沿着中心线裁剪
    pos = 0
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

            loc_record.append(s)
            flag_record[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = 0
            pos = pos + 1

    return data_patch_list, label_patch_list, loc_record


class Get_patch_from_pre: ## pre_label.nii.gz   label
    def __init__(self, data_path, label_path, pre_path, save_path, patch_size,data_type,prior_name='pre_label.nii.gz'):
        '''
        :param data_path: 真实图像路径 data_path/id/img.nii.gz 输入的是data_path
        :param label_path: 真实标签路径：label_path/id/label.nii.gz 输入的是label_path
        :param pre_path: 预测标签路径：pre_path/id/pre_label.nii.gz
        :param save_path: 储存patch合集的位置
        :param patch_size: patch大小为int
        :param data_type: str 数据类型，train or valid
        '''

        self.data_path = data_path
        self.label_path = label_path
        self.pre_path = pre_path
        self.prior_name=prior_name
        self.patch_size = patch_size
        self.data_type = data_type

        self.save_img_path = os.path.join(save_path, 'patch_%d' % patch_size, '%s_img_patch' % data_type)
        self.save_label_path = os.path.join(save_path, 'patch_%d' % patch_size, '%s_label_patch' % data_type)
        self.save_record_path = os.path.join(save_path, 'patch_%d' % patch_size, 'csv_patch_record')

        os.makedirs(self.save_img_path,exist_ok=True)
        os.makedirs(self.save_label_path,exist_ok=True)
        os.makedirs(self.save_record_path,exist_ok=True)

    def run(self, i):
        df = pd.DataFrame(columns=['x', 'y', 'z'])
        d_path = os.path.join(self.data_path, i, 'img.nii.gz')
        l_path = os.path.join(self.label_path, i, 'label.nii.gz')
        p_path=os.path.join(self.pre_path,i,self.prior_name)

        d_nii = nib.load(d_path)
        l_nii = nib.load(l_path)
        p_nii=nib.load(p_path)

        p_label = p_nii.get_fdata()
        if p_label.max()==255:
            p_label=p_label/255
        label=l_nii.get_fdata()
        img = d_nii.get_fdata()

        img_list, label_list, loc_record = get_patch(img,label, p_label, self.patch_size)

        ID = i
        affine = d_nii.affine

        for index, p in enumerate(img_list):
            nib.save(nib.Nifti1Image(p, affine), os.path.join(self.save_img_path, ID + '_%d.nii.gz' % index))
            nib.save(nib.Nifti1Image(label_list[index], affine),
                     os.path.join(self.save_label_path, ID + '_%d.nii.gz' % index))
            df.loc[ID + '_%d.nii.gz' % index] = [loc_record[index][0], loc_record[index][1], loc_record[index][2]]
            df.to_csv(os.path.join(self.save_record_path, ID + '.csv'))


