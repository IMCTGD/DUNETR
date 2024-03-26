import numpy as np
import pandas as pd
import os
import nibabel as nib


def add_patch(img, patch, s, p_size):
    e = np.array(s) + p_size
    p1 = img[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
    p2 = np.maximum(p1, patch)
    img[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = p2
    return img


class Recover_patch():
    def __init__(self, patch_size, patch_pre, record_csv_path, save_pre_path, data_path,save_file_name='pre_label.nii.gz'):
        '''
        将patch整合为原来3D图像的大小
        :param patch_size: patch的大小
        :param patch_pre: 预测patch的文件目录
        :param record_csv: 保存patch位置的文件目录 record_csv_path/id.csv
        :param save_pre_path: 保存生成3D图像的位置 data_path/img.nii.gz
        :param data_path: 原数据的目录
        '''

        self.patch_size = patch_size
        self.patch_pre = patch_pre
        self.record_csv_path = record_csv_path
        self.save_pre_path = save_pre_path
        self.data_path = data_path
        self.save_file_name=save_file_name

    def run_recover(self, id):

        os.makedirs(self.save_pre_path, exist_ok=True)
        # print('id:', id)
        o_data = nib.load(os.path.join(self.data_path, id, 'img.nii.gz')).get_data()

        img = np.zeros_like(o_data, dtype=np.float)
        flag = 0
        p_affine = 0
        record = pd.read_csv(os.path.join(self.record_csv_path, id + '.csv'), index_col=0)

        for i in record.index:
            p_nii = nib.load(os.path.join(self.patch_pre, i))
            if flag == 0:
                flag = 1
                p_affine = p_nii.affine
            s = (record.loc[i]['x'], record.loc[i]['y'], record.loc[i]['z'])
            img = add_patch(img, p_nii.get_data(), s, self.patch_size)
        img_bina = np.copy(img)
        img_bina[img > 0.5] = 1
        img_bina[img <= 0.5] = 0
        os.makedirs(os.path.join(self.save_pre_path, id), exist_ok=True)
        # nib.save(nib.Nifti1Image(img, p_affine),os.path.join(self.save_pre_path, id, self.patch_size))
        nib.save(nib.Nifti1Image(img_bina, p_affine),os.path.join(self.save_pre_path, id, self.save_file_name))
