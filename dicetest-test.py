#  开发人员：    骆根强
#  开发时间：    2022/11/24 22:36
#  功能作用：    未知

from tqdm import tqdm
import nibabel as nib
import os
from utils.utils import reshape_img
from utils.parallel import parallel
import config
import torch
import argparse
import os
import numpy as np
from data.Image_loader import CoronaryImage
from utils.utils import get_csv_split
from torch.utils.data import DataLoader
import torch.optim as optim
from model.loss import DiceLoss,DiceLoss_v1
import torch.nn as nn
import time
from model.FCN import FCN_Gate, FCN
import multiprocessing
import yaml
import re
from utils.Calculate_metrics import Cal_metrics
import pandas as pd
from transformer_seg import SETRModel
from datetime import datetime
import torch.nn.functional as F
from unetr import UNETR
import config
import time
from data.data_utils import get_loader
from monai.inferers import sliding_window_inference
# from utils.crf import DenseCRF
from model.CNN_model import Unet, Unet_Patch, NestedUNet3d
from DUNETR import DUNETR
from SwinDUNETR import SwinDUNETR
from monai.networks.nets import SwinUNETR
from model.unet3d import UNet3D



def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

# setup_postprocessor = DenseCRF(
#         iter_max=10,
#         pos_xy_std=1,
#         pos_w=3,
#         bi_xy_std=67,
#         bi_rgb_std=3,
#         bi_w=4,
#     )


def inference(model, valid_loader, device, save_img_path):
    model.eval()
    dice_list_sub = []
    for batch in (valid_loader):
        img, label = batch['image'].float(), batch['label'].float()
        img, label = img.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(img)
        outputs = torch.sigmoid(outputs)

        # import torchvision
        # from torchvision import transforms
        # cor = torchvision.transforms.CenterCrop(64)
        # for i in range(outputs.shape[1]):
        #     for j in range(outputs.shape[2]):
        #         torchvision.utils.save_image(cor(outputs[0,i,j,:,:]), f'./out4/ima_{i}_{j}.png')
        #
        # print(qwe)
        outputs = outputs.squeeze(1)
        pre = outputs.cpu().detach().numpy()
        pre = pre.transpose((0, 2, 3, 1))
        ## 保存文件
        ID = batch['image_index']
        affine = batch['affine']
        img_size = batch['image_size']
        os.makedirs(save_img_path, exist_ok=True)
        batch_save(ID, affine, pre, img_size, save_img_path)

        ## 后面编写的测试dice
        val_outputs = sliding_window_inference(img, (input_size[2], input_size[0], input_size[1]), 4, model, overlap=args.infer_overlap)
        val_labels = label.cpu().numpy()
        ## 格式转换
        val_outputs = val_outputs.cpu().detach().numpy()
        val_outputs[val_outputs >= 0.5] = 1
        val_outputs[val_outputs < 0.5] = 0

        ## 计算dice系数
        organ_Dice = dice(val_outputs, val_labels)
        dice_list_sub.append(organ_Dice)
        print(f"\033[31mTest-ID:{batch['image_index']}  || Dice: {organ_Dice} \033[0m")
    mean_dice = np.mean(dice_list_sub)
    print("\033[31mACC_Dice:%f  \033[0m" % (mean_dice))

def batch_save(ID, affine, pre, img_size, save_img_path):
    batch_size = len(ID)
    save_list = [save_img_path] * batch_size
    parallel(save_picture, pre, affine, img_size, save_list, ID, thread=True)

def save_picture(pre, affine, img_size, save_name, id):
    pre_label = pre
    pre_label[pre_label >= 0.5] = 1
    pre_label[pre_label < 0.5] = 0
    pre_label = reshape_img(pre_label, img_size.numpy())
    os.makedirs(os.path.join(save_name, id), exist_ok=True)
    ## 保存医学图像信息, pre_label的格式是【128,128,64】，得把3通道放到最后的位置
    nib.save(nib.Nifti1Image(pre_label, affine), os.path.join(save_name, id + '/pre_label.nii.gz'))

def args_input():
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--gpu_index', type=str, default='0')
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument("--data_dir", default="F:/data/data_1100/", type=str, help="dataset directory")
    p.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")

    p.add_argument('--channel', type=int, default=4)        ## 选择通道数
    p.add_argument('--rl', type=int, default=2)             ## 选择分辨率

    p.add_argument('--model', type=str, default='UNet3D')   # DUNETR、UNETR、FCN_AG_scr、Unet++、UNet3D
    p.add_argument('--fold', type=int, default=1)           ## 选择哪个数字作为测试集
    p.add_argument('--load_num', type=int, default=56)     ## 0 代表不加载参数,    1加载
    p.add_argument('--batch_size', type=int, default=1)     ## batch_size 的大小
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--pools', type=int, default=2)

    p.add_argument('--loss', type=str, default='Dice')      ## 选择损失函数：Dice、Dice_dilation
    p.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    p.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    p.add_argument('--flip_prob', type=float, default=0)
    p.add_argument('--rotate_prob', type=float, default=0)

    ## data
    p.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    p.add_argument("--a_max", default=300.0, type=float, help="a_max in ScaleIntensityRanged")
    p.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    p.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")

    p.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    p.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    p.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")

    p.add_argument("--roi_x", default=48, type=int, help="roi size in x direction")  ##  随机裁剪XYZ大小
    p.add_argument("--roi_y", default=48, type=int, help="roi size in y direction")
    p.add_argument("--roi_z", default=48, type=int, help="roi size in z direction")

    p.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    p.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")  ## 镜像旋转
    p.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")  ## 角度旋转
    p.add_argument("--RandScaleIntensityd_prob", default=0.2, type=float, help="RandScaleIntensityd aug probability")
    p.add_argument("--RandShiftIntensityd_prob", default=0.2, type=float, help="RandShiftIntensityd aug probability")
    p.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    p.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")

    p.add_argument("--distributed", action="store_true", help="start distributed training")
    p.add_argument("--workers", default=1, type=int, help="number of workers")  ## 多线程
    p.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
    p.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
    p.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")

    ## VIT
    p.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    p.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
    p.add_argument("--mlp_dim", default=1024, type=int, help="mlp dimention in ViT encoder")
    p.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")  ## 多少个TF
    p.add_argument("--feature_size", default=8, type=int, help="feature size dimention")
    p.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    p.add_argument("--out_channels", default=1, type=int, help="number of output channels")

    return p.parse_args()

if __name__ == '__main__':
    ##参数解析
    args = args_input()
    gpu_index = args.gpu_index
    config_file = args.config_file
    k = args.fold
    load_num = args.load_num
    batch_size = args.batch_size
    model_name = args.model
    channel = args.channel
    resolution = args.rl
    pool_nums = args.pools
    num_workers = args.num_workers
    loss_name = args.loss
    result_path = r'result/Direct_seg'

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    # torch.cuda.set_device(0)
    # torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if resolution == 1:
        resolution_name = 'High_resolution'
        input_size = [512, 512, 256]
    elif resolution == 2:
        resolution_name = 'Mid_resolution'
        input_size = [256, 256, 128]
    elif resolution == 3:
        resolution_name = 'Low_resolution'
        input_size = [128, 128, 64]
    else:
        raise ValueError("没有该级别的分辨率")

    # 从config.yaml里面读取参数
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        learning_rate = config['General_parameters']['lr']
        train_path = config['General_parameters']['data_path']
        train_path = train_path.replace('chosen_dia', 'test-data')

        valid_path = config['General_parameters']['data_path']
        valid_path = train_path.replace('chosen_dia', 'test-data')
        csv_path = config['General_parameters']['csv_path']

    parameter_record = resolution_name + '_%d_' % channel + loss_name

    print('model: %s || parameters: %s || %d_fold' % (model_name, parameter_record, k))

    # 读取参数配置文件
    model_save_path = r'%s/%s/%s/fold_%d/model_save' % (result_path, model_name, parameter_record, k)
    save_label_path = r'%s/%s/%s/fold_%d/test_label' % (result_path, model_name, parameter_record, k)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)
    ID_list = get_csv_split('./data_list/test-80.csv', 6)  ## 把数据集划分成训练和测试，规律是按K_flod里的K值来分，这里是list的文件夹名字，1为测试。其余为训练

    # # 网络模型
    # net = FCN_Gate(channel).to(device)
    # # 网络模型
    if model_name == "DUNETR":
        net = DUNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(input_size[2], input_size[0], input_size[1]),
            # feature_size=10,                 ## 卷积通道
            feature_size=args.feature_size,  ## 卷积通道
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            # dropout_rate=args.dropout_rate,
        ).to(device)
    elif model_name == "SwinDUNETR":
        net = SwinDUNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(input_size[2], input_size[0], input_size[1]),
            # feature_size=10,                 ## 卷积通道
            feature_size=args.feature_size,  ## 卷积通道
            # hidden_size=args.hidden_size,
            # mlp_dim=args.mlp_dim,
            # num_heads=args.num_heads,
            # pos_embed='perceptron',
            # norm_name='instance',
            # conv_block=True,
            # res_block=True,
            dropout_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        ).to(device)
    elif model_name == "UNETR":
        net = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(input_size[2], input_size[0], input_size[1]),
            # feature_size=10,                 ## 卷积通道
            feature_size=args.feature_size,  ## 卷积通道
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            # dropout_rate=args.dropout_rate,
        ).to(device)
    elif model_name == "SwinUNETR":
        net = SwinUNETR(
            img_size=(input_size[2], input_size[0], input_size[1]),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        ).to(device)
    elif model_name == "FCN_AG_scr":
        net = FCN_Gate(channel).to(device)
    elif model_name == "Unet++":
        net = NestedUNet3d(1, 1, 6).to(device)
    elif model_name == "UNet3D":
        net = UNet3D(1,1).to(device)
        print("UNet3D")
    else:
        raise ValueError("模型错误")

    model_list = os.listdir(model_save_path)

    # net.load_state_dict(torch.load(model_save_path + '/net_scr_%s_%d.pkl' % (model_name, load_num)))
    net.load_state_dict(torch.load(model_save_path + '/net_scr_%s_%d.pkl' % (model_name, load_num), map_location='cpu'))

    net_opt = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = DiceLoss()

    # 推断
    # # 数据加载
    valid_set = CoronaryImage(valid_path, valid_path, ID_list['valid'], input_size)
    valid_infer_loader = DataLoader(valid_set, 1, num_workers=num_workers, shuffle=False)
    # #
    # valid_set = get_loader(valid_path, valid_path, ID_list['valid'], input_size, args, False)
    # valid_infer_loader = DataLoader(valid_set, 1, num_workers=num_workers, shuffle=False)

    print('now inference..............')
    inference(net, valid_infer_loader, device, save_label_path)

    # 计算最后的dice
    print('now calculate dice...........')
    CD = Cal_metrics(os.path.join(save_label_path), valid_path, 0)
    p = multiprocessing.Pool(pool_nums)
    result = p.map(CD.calculate_dice, ID_list['valid'])
    print('-炮打鬼-' * 20)
    p.close()
    p.join()

    # 保存结果
    record_dice = {}
    record_dice['id'] = ID_list['valid']
    result = np.array(result)
    record_dice['dice'] = result[:, 0]
    record_dice['ahd'] = result[:, 1]
    record_dice['hd'] = result[:, 2]
    record_dice = pd.DataFrame(record_dice)
    record_dice.to_csv(r'%s/%s/%s/fold_%d/test-result.csv' % (result_path, model_name, parameter_record, k), index=False)
