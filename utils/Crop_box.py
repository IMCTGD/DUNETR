import numpy as np
import os
import nibabel as nib

def get_crop(pre_label, img, label,enhance):
    '''
    根据粗分割，统计出一个固定的边界框
    :param pre_label: 粗分割预测图像 3D array
    :param img: CT图像 3D array
    :param label: 标签 3D array
    :param enhance: 增强图像 3D array
    :return:
    '''
    x, y, z = np.where(pre_label == 1)
    img_shape = img.shape

    x_max = (x.max() + 5) if (x.max() + 5) <= img_shape[0] else img_shape[0]
    y_max = (y.max() + 5) if (y.max() + 5) <= img_shape[1] else img_shape[1]
    z_max = (z.max() + 5) if (z.max() + 5) <= img_shape[2] else img_shape[2]

    x_min = x.min() - 5 if (x.min() - 5) >= 0 else 0
    y_min = y.min() - 5 if (y.min() - 5) >= 0 else 0
    z_min = z.min() - 5 if (z.min() - 5) >= 0 else 0

    loc_max = [x_max, y_max, z_max]
    loc_min = [x_min, y_min, z_min]

    img_crop = img[loc_min[0]:loc_max[0], loc_min[1]:loc_max[1], loc_min[2]:loc_max[2]]
    label_crop = label[loc_min[0]:loc_max[0], loc_min[1]:loc_max[1], loc_min[2]:loc_max[2]]
    enhance_crop = enhance[loc_min[0]:loc_max[0], loc_min[1]:loc_max[1], loc_min[2]:loc_max[2]]
    return img_crop, label_crop, enhance_crop, loc_min, loc_max


class Crop_pre():
    def __init__(self,coarse_path,img_path,save_path,enhance_path,pre_file_name='pre_label.nii.gz'):
        '''
        :param coarse_path: 预测标签路径 coarse_path/id/label.nii.gz
        :param img_path: 真实图像路径 data_path/id/img.nii.gz
        :param save_path: 真实标签路径 label_path/id/label.nii.gz
        :param enhance_path: 增强图像路径 enhance_path/id/label.nii.gz
        '''
        self.coarse_path=coarse_path
        self.img_path=img_path
        self.save_path=save_path
        self.enhance_path=enhance_path
        self.pre_file_name=pre_file_name

        os.makedirs(save_path,exist_ok=True)

    def run_crop(self,i):
        print(i)
        pre_nii = nib.load(os.path.join(self.coarse_path, i, self.pre_file_name))
        pre = pre_nii.get_fdata()

        img_nii = nib.load(os.path.join(self.img_path, i, 'img.nii.gz'))
        img=img_nii.get_fdata()
        label = nib.load(os.path.join(self.img_path, i, 'label.nii.gz')).get_fdata()
        # print('**' * 20)
        # print(os.path.join(self.enhance_path, i, 'frangi.nii.gz'))
        # print(qw)
        enhance = nib.load(os.path.join(self.enhance_path, i, 'frangi.nii.gz')).get_fdata()

        img_crop, label_crop,enhance_crop, l_min, l_max = get_crop(pre, img, label,enhance)
        s_path = os.path.join(self.save_path, i)

        os.makedirs(s_path, exist_ok=True)
        nib.save(nib.Nifti1Image(img_crop, pre_nii.affine,header=img_nii.header), os.path.join(s_path, 'img.nii.gz'))
        nib.save(nib.Nifti1Image(label_crop, pre_nii.affine), os.path.join(s_path, 'label.nii.gz'))
        nib.save(nib.Nifti1Image(enhance_crop, pre_nii.affine), os.path.join(s_path, 'frangi.nii.gz'))

        return l_min + l_max


class Recover_Crop:
    def __init__(self,pre_path,data_path,crop_dict,pre_file_name='pre_crop.nii.gz',save_file_name='pre_label.nii.gz'):
        '''
        :param coarse_path: 预测标签路径 coarse_path/id/label.nii.gz
        :param img_path: 真实图像路径 data_path/id/img.nii.gz
        :param save_path: 真实标签路径 label_path/id/label.nii.gz
        :param enhance_path: 增强图像路径 enhance_path/id/label.nii.gz
        '''
        self.data_path=data_path
        self.pre_path=pre_path
        self.crop_dict=crop_dict
        self.pre_file_name=pre_file_name
        self.save_file_name=save_file_name

    def run(self,i):
        print(i)
        i_list = self.crop_dict['ID']
        index = i_list.index(i)
        l_loc = self.crop_dict['box'][index]

        pre_nii = nib.load(os.path.join(self.pre_path, i, self.pre_file_name))
        pre = pre_nii.get_fdata()
        img=nib.load(os.path.join(self.data_path, i, 'img.nii.gz')).get_fdata()

        pre_label=np.zeros_like(img)
        pre_label[l_loc[0]:l_loc[3],l_loc[1]:l_loc[4],l_loc[2]:l_loc[5]]=pre

        nib.save(nib.Nifti1Image(pre,pre_nii.affine),os.path.join(self.pre_path, i, self.save_file_name))


