#  开发人员：    骆根强
#  开发时间：    2022/11/15 11:37
#  功能作用：    未知
from scipy.ndimage.interpolation import zoom
import torch
import argparse
import os
import numpy as np
from data.Image_loader import CoronaryImage
from utils.utils import get_csv_split
from torch.utils.data import DataLoader
import torch.optim as optim
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
from DUNETR import DUNETR
from SwinDUNETR import SwinDUNETR
from u2net import U2NET
# from monai.losses import DiceCELoss ##, DiceLoss
from model.loss import DiceLoss,DiceLoss_v1
from torch.utils.data.distributed import DistributedSampler
#from TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from monai.networks.nets import SwinUNETR
from model.unet3d import UNet3D


# dice_loss = DiceLoss()
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, criterion):
    loss0 = criterion(d0,labels_v)
    loss1 = criterion(d1,labels_v)
    loss2 = criterion(d2,labels_v)
    loss3 = criterion(d3,labels_v)
    loss4 = criterion(d4,labels_v)
    loss5 = criterion(d5,labels_v)
    loss6 = criterion(d6,labels_v)

    loss = 0.5*loss0 + 0.1*loss1 + 0.1*loss2 + 0.1*loss3 + 0.1*loss4 + 0.05*loss5 + 0.05*loss6
    # loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("\033[31m All: %3f\033[0m , l0:\033[31m  %3f\033[0m , l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f"%
          (loss.data.item(), loss0.data.item(), loss1.data.item(), loss2.data.item(),
           loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))
    return loss, loss0

def train(model, criterion, train_loader, opt, device, e):
    model = model.to(device)
    model.train()
    train_sum = 0
    train_sum2 = 0
    ite_num = 0
    for j, batch in enumerate(train_loader):
        img, label = batch['image'].float(), batch['label'].float()
        img, label = img.to(device), label.to(device)
        # a,b = torch.unique(label[0],return_counts=True)
        # a = img.squeeze(0).squeeze(0).cpu().numpy()
        ite_num = ite_num + 1

        outputs = model(img)
        print(awe)
        opt.zero_grad()
        loss = criterion(outputs, label)
        print('{}  Epoch {:>3d}/{:<3d} | Step {:>3d}/{:<3d} | train loss {:.4f}'
              .format(datetime.now(), e, args.epochs, j, len(train_loader), 1-loss.item()))
        train_sum += loss.item()
        loss.backward()
        opt.step()
    print('average_train_loss: %f' % (train_sum / len(train_loader)))
    return train_sum / len(train_loader)

def valid(model, criterion, valid_loader, device, e):
    model.eval()
    valid_sum = 0
    for j, batch in enumerate(valid_loader):
        img, label = batch['image'].float(), batch['label'].float()
        img, label = img.to(device), label.to(device)

        with torch.no_grad():
            # outputs, d1, d2, d3, d4, d5, d6 = model(img)
            outputs= model(img)
            loss = criterion(outputs, label)
        valid_sum += loss.item()
        print('{} Epoch {:>3d}/{:<3d}  |Step {:>3d}/{:<3d}  | valid loss {:.4f}'
              .format(datetime.now(), e, args.epochs, j, len(valid_loader), 1-loss.item()))

    return valid_sum / len(valid_loader)

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool3d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3, 4)) / weit.sum(dim=(2, 3, 4))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3, 4))
    union = ((pred + mask)*weit).sum(dim=(2, 3, 4))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def args_input():
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--gpu_index', type=str, default='0')
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument("--data_dir", default="F:/data/data_1100/", type=str, help="dataset directory")

    p.add_argument('--fold', type=int, default=1)           ## 选择哪个数字作为测试集
    p.add_argument('--is_train', type=int, default=1)       ## 是否训练,1代表训练

    p.add_argument('--load_num', type=int, default=0)       ## 0 代表不加载参数,    1加载
    p.add_argument('--is_inference', type=int, default=1)   ## 是否算出结果,1代表处结果

    p.add_argument('--model', type=str, default='UNETR')   # DUNETR、SwinDUNETR、SwinUNETR、UNet3D
    p.add_argument('--channel', type=int, default=4)        ##
    p.add_argument('--rl', type=int, default=3)             ## 选择分辨率
    p.add_argument('--batch_size', type=int, default=1)     ## batch_size 的大小
    p.add_argument('--num_workers', type=int, default=0)    ## 线程数
    p.add_argument('--pools', type=int, default=4)
    p.add_argument('--lr_rate', type=int, default=0.001)

    p.add_argument('--loss', type=str, default='Dice')  ## 选择损失函数：Dice、Dice_dilation
    p.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    p.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    p.add_argument('--flip_prob', type=float, default=0)
    p.add_argument('--rotate_prob', type=float, default=0)

    ## data
    # p.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    p.add_argument("--a_min", default=10.0, type=float, help="a_min in ScaleIntensityRanged")
    p.add_argument("--a_max", default=500.0, type=float, help="a_max in ScaleIntensityRanged")
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
    p.add_argument("--feature_size", default=12, type=int, help="feature size dimention")
    p.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    p.add_argument("--out_channels", default=1, type=int, help="number of output channels")

    p.add_argument('--epochs', type=int, default=200)
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
    is_train = args.is_train
    loss_name=args.loss
    epochs=args.epochs
    result_path = r'result/Direct_seg'
    lr_rate = args.lr_rate

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    # torch.cuda.set_device(0)
    # torch.backends.cudnn.enabled = True
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_device = torch.cuda.current_device()
    print("Device:", torch.cuda.get_device_name(current_device))

    if resolution == 1:
        resolution_name = 'High_resolution'
        input_size = [512, 512, 256]
        print('选着的分辨率为： ', input_size)
    elif resolution == 2:
        resolution_name = 'Mid_resolution'
        input_size = [256, 256, 128]
        print('选着的分辨率为： ', input_size)
    elif resolution == 3:
        resolution_name = 'Low_resolution'
        # input_size = [128, 128, 64]
        # input_size = [96, 96, 96]
        input_size = [64, 64, 32]
        print('选着的分辨率为： ', input_size)
    else:
        raise ValueError("没有该级别的分辨率")

    # 从config.yaml里面读取参数
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        #learning_rate = config['General_parameters']['lr']
        learning_rate = lr_rate
        train_path = config['General_parameters']['data_path']
        valid_path = config['General_parameters']['data_path']
        csv_path = config['General_parameters']['csv_path']

    parameter_record = resolution_name + '_%d_' % channel+ loss_name

    print('model: %s || parameters: %s || %d_fold' % (model_name, parameter_record, k))

    # 读取参数配置文件
    model_save_path = r'%s/%s/%s/fold_%d/model_save' % (result_path, model_name, parameter_record, k)
    save_label_path = r'%s/%s/%s/fold_%d/pre_label' % (result_path, model_name, parameter_record, k)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)
    ID_list = get_csv_split(csv_path, k)        ## 把数据集划分成训练和测试，规律是按K_flod里的K值来分，这里是list的文件夹名字，1为测试。其余为训练

    # # # 数据加载
    train_set = CoronaryImage(train_path, train_path, ID_list['train'], input_size)
    valid_set = CoronaryImage(valid_path, valid_path, ID_list['valid'], input_size)
    ##### DataLoader出来的是一个字典——键为：image、label、affine(参考空间)、image_index、image_size
    train_loader = DataLoader(train_set, batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size, num_workers=num_workers, shuffle=False)

    # train_set = get_loader(train_path, train_path, ID_list['train'], input_size, args, True)
    # valid_set = get_loader(train_path, train_path, ID_list['valid'], input_size, args, False)
    # train_loader = DataLoader(train_set, batch_size, num_workers=num_workers, shuffle=False)
    # valid_loader = DataLoader(valid_set, batch_size, num_workers=num_workers, shuffle=False)

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
            feature_size=args.feature_size,    ## 卷积通道
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
    elif model_name == "UNet3D":
        net = UNet3D(1,1).to(device)
        print("UNet3D")
    else:
        raise ValueError("模型错误")

    # net = nn.DataParallel(net)

    model_list = os.listdir(model_save_path)
    net_opt = optim.Adam(net.parameters(), lr=learning_rate)

    # 是否加载网络模型
    if load_num == 0:
        print('-'*20, "\033[31m重新训练参数\033[0m", '-'*20)
        load_num = load_num + 1
        for m in net.modules():
            if isinstance(m, (nn.Conv3d)):
                nn.init.orthogonal(m.weight)
    else:
        print('-'*20, "\033[31m加载参数\033[0m", '-'*20)
        net.load_state_dict(torch.load(model_save_path + '/net_scr_%s_%d.pkl' % (model_name, load_num)))
        load_num = load_num + 1

    if loss_name=='Dice':
        criterion = DiceLoss()
        # criterion = nn.BCELoss(size_average=True)
        # criterion = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True,
        #                        smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr)
        # criterion = structure_loss
    elif loss_name=='Dice_dilation':
        criterion = DiceLoss_v1()
    else:
        raise ValueError("No loss")

    train_loss_set = []
    valid_loss_set = []
    epoch_list = []

    print('+'*40)
    print('='*10, '使用的设备是：', format(device), '='*10)
    para_total = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('='*10, '参数量为：{:0>4}_{:0>4}'.format(int(para_total/10000),int(para_total%10000)),'='*10)
    print('+'*40)
    valid_loss = 1
    # 训练
    if is_train == 1:
        for e in range(load_num, epochs+1):
            start = time.time()
            print(f"============={model_name}_train=============")
            train_loss = train(net, criterion, train_loader, net_opt, device, e)
            train_loss_set.append(1-train_loss)
            if e % 1 ==0:
                print(f"============={model_name}_valid=============")
                valid_loss = valid(net, criterion, valid_loader, device, e)
                print('==' * 20 + "\033[31mTIME:%f S || Valid_loss:%f \033[0m" % (int(time.time() - start), 1-valid_loss))
            valid_loss_set.append(1-valid_loss)
            epoch_list.append(e)
            # print('=='*20+"\033[31mTIME:%f S || train_loss:%f || valid_loss:%f\033[0m" % (int(time.time()-start),train_loss, valid_loss))
            print('==' * 20 + "\033[31mTIME:%f S || train_loss:%f \033[0m" % (int(time.time() - start), 1-train_loss))
            torch.save(net.state_dict(), model_save_path + '/net_scr2_%s_%d.pkl' % (model_name,e))
            record = dict()
            record['epoch'] = epoch_list
            record['train_loss'] = train_loss_set
            record['valid_loss'] = valid_loss_set
            record = pd.DataFrame(record)
            record_name = time.strftime(f"{k}-%Y_%m_%d_%H", time.localtime())
            csvname = f'{record_name}.csv'
            record.to_csv(f'{save_label_path}/{csvname}', index=False)

            # ##############################################################
            # print("=============valid=============")
            # valid_loss = valid(net, criterion, valid_loader, device, e)
            # valid_loss_set.append(valid_loss)
            # epoch_list.append(e)
            # print('==' * 20 + "\033[31mTIME:%f S || train_loss:%f || valid_loss:%f\033[0m" % (
            # int(time.time() - start), train_loss, valid_loss))
            # torch.save(net.state_dict(), model_save_path + '/net_scr_%s_%d.pkl' % (model_name, e))
            # record = dict()
            # record['epoch'] = epoch_list
            # record['train_loss'] = train_loss_set
            # record['valid_loss'] = valid_loss_set
            # record = pd.DataFrame(record)
            # record_name = time.strftime("%Y_%m_%d_%H", time.localtime())
            # csvname = f'{record_name}.csv'
            # record.to_csv(f'./{csvname}', index=False)






