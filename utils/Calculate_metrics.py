import nibabel
import numpy as np
import os
from skimage.measure import label
import SimpleITK as sitk
import torch
import torch.nn as nn
from scipy.ndimage.interpolation import zoom

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

def dice_coef(y_true, y_pred):
    gt = sitk.GetImageFromArray(y_true, isVector=False)
    my_mask = sitk.GetImageFromArray(y_pred, isVector=False)

    dice_dist = sitk.LabelOverlapMeasuresImageFilter()
    dice_dist.Execute(gt > 0.5, my_mask > 0.5)
    dice = dice_dist.GetDiceCoefficient()
    return dice

def HausdorffDistance(y_true, y_pred, spaceing):
    gt = sitk.GetImageFromArray(y_true, isVector=False)
    mask = sitk.GetImageFromArray(y_pred, isVector=False)
    gt.SetSpacing(spaceing)
    mask.SetSpacing(spaceing)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(gt > 0.5, mask > 0.5)
    AvgHD = hausdorffcomputer.GetAverageHausdorffDistance()
    HD = hausdorffcomputer.GetHausdorffDistance()
    return AvgHD, HD


def get_region_num(img, get_num):
    '''
    :param img: 输入标签图像
    :param get_num: 保留最大连通域的个数
    :return: 提取连通域后的numpy数组
    '''
    conn_img = label(img, connectivity=3)
    i_max = conn_img.max()
    if get_num > i_max:
        get_num = i_max
    area_list = []
    for i in np.arange(1, i_max + 1):
        area_list.append(np.sum(conn_img == i))
    index = np.argsort(-np.array(area_list)) + 1
    record = np.zeros_like(img)
    for i in np.arange(get_num):
        record[conn_img == index[i]] = 1
    return record

def reshape_img(img, output_shape, order=0):
    '''
    :param img: 3D array
    :param output_shape: 输出图像大小
    :param order: 插值方式
    :return: 插值后的图像
    '''
    s = (output_shape[0] / img.shape[0], output_shape[1] / img.shape[1], output_shape[2] / img.shape[2])
    # s = (output_shape[0] / img.shape[0], output_shape[1] / img.shape[1], 1)    ## 不改变Z轴
    new_img = zoom(img, zoom=s, order=order)
    return new_img

class Cal_metrics:
    '''
    计算指标多进程版
    预测标签的目录为id/pre_label.nii.gz
    真实标签的目录为id/(label.nii.gz,image.nii.gz)
    '''
    def __init__(self, pre_path, true_path, con_num=0 ,pre_label_name='pre_label.nii.gz',is_use_prob=True):
        '''
        :param pre_path:  the path of prediction
        :param true_path: the path of true label
        :param con_num: the number of max connected domain would be reserved, 如果为0将保留所有连通域
        :param is_use_prob: 是否进行二值化操作
        '''
        self.pre_path = pre_path
        self.true_path = true_path
        self.con_num= con_num
        self.is_use_prob = is_use_prob
        self.pre_label_name=pre_label_name
        self.dice_list_sub = []

    def calculate_dice(self, i):
        '''
        :param i: id号，id号必须存在于预测路径以及真实路径
        :return:
        '''
        pre_nii = nibabel.load(os.path.join(self.pre_path, i, self.pre_label_name))
        pre = pre_nii.get_fdata()

        if self.is_use_prob:
            pre[pre >= 0.5] = 1
            pre[pre < 0.5] = 0

        if self.con_num!=0:
            pre = get_region_num(pre, self.con_num)

        true = nibabel.load(os.path.join(self.true_path, i, 'label.nii.gz')).get_fdata()
        data_nii = nibabel.load(os.path.join(self.true_path, i, 'img.nii.gz'))

        header = data_nii.header
        spacing = header.get_zooms()
        spacing = tuple([float(spacing[0]), float(spacing[1]), float(spacing[2])])

        ## 低分辨率
        # true = reshape_img(true, [128, 128, 64])
        # pre = reshape_img(pre, [128, 128, 64])
        ## 中分辨率
        # true = reshape_img(true, [256, 256, 128])
        # pre = reshape_img(pre, [256, 256, 128])
        ## 高分辨率
        # true = reshape_img(true, [512, 512, 256])
        # pre = reshape_img(pre, [512, 512, 256])

        max_dice = dice_coef(true, pre)
        # max_dice = dice(true, pre)

        self.dice_list_sub.append(max_dice)
        mean_dice = np.mean(self.dice_list_sub)

        ahd, hd = HausdorffDistance(true, pre, spacing)
        print(i,' dice:%f hd:%f Acc_dice:%f'%(max_dice, hd, mean_dice))
        # print(i,' dice:%f ' % (max_dice))

        return max_dice, ahd, hd
