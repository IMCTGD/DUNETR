from skimage.filters import ridges
import nibabel as nib
import numpy as np
import os
import multiprocessing

class Frangi():
    def __init__(self,img_path,enhance_save_path,enhance_name='frangi.nii.gz'):
        self.img_path=img_path
        self.enhance_name=enhance_name
        self.enhance_save_path=enhance_save_path

    def run_enhance(self,i):
        print(i)
        i_path = os.path.join(self.img_path, i, 'img.nii.gz')
        img_nii = nib.load(i_path)
        img = img_nii.get_fdata()
        # print('！！' * 20)
        # print(os.path.join(self.enhance_save_path, i))
        # print(qw)
        # print('!!!!!!!!!!!!!!!!!!!!!!')
        # print(img.shape)
        # print(qw)
        img = ridges.frangi(img, sigmas=range(1, 5, 2), black_ridges=False) ## 利用算法 多尺度血管增强滤波
        os.makedirs(os.path.join(self.enhance_save_path, i), exist_ok=True)
        nib.save(nib.Nifti1Image(img, img_nii.affine), os.path.join(self.enhance_save_path, i, self.enhance_name))
        print(i,':done')