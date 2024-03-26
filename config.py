#  开发人员：    骆根强
#  开发时间：    2022/11/22 17:55
#  功能作用：    未知

import argparse

p = argparse.ArgumentParser(description='cmd parameters')
p.add_argument('--gpu_index', type=str, default='0')
p.add_argument('--config_file', type=str, default='config/config.yaml')
# p.add_argument("--data_dir", default="./config/", type=str, help="dataset directory")
p.add_argument("--data_dir", default="F:/data/data_1100/", type=str, help="dataset directory")
p.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
p.add_argument('--Direct_parameter', type=str, default='Low_resolution_4_Dice')## Mid_resolution_4_Dice

p.add_argument('--fold', type=int, default=1)       ## 选择哪个数字作为测试集
p.add_argument('--is_train', type=int, default=1)   ## 是否训练,1代表训练

p.add_argument('--load_num', type=int, default=18)   ## 0 代表不加载参数,    1加载
p.add_argument('--is_inference', type=int, default=1)   ## 是否算出结果,1代表处结果

p.add_argument('--model', type=str, default='U2NET')#DUNETR
p.add_argument('--channel', type=int, default=4)        ##
p.add_argument('--rl', type=int, default=3)            ## 选择分辨率
p.add_argument('--batch_size', type=int, default=1)     ## batch_size 的大小
p.add_argument('--num_workers', type=int, default=4)
p.add_argument('--pools', type=int, default=4)
p.add_argument('--patch_size', type=int, default=128)


p.add_argument('--loss',type=str,default='Dice')    ## 选择损失函数：Dice、Dice_dilation
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

p.add_argument("--roi_x", default=48, type=int, help="roi size in x direction")        ##  随机裁剪XYZ大小
p.add_argument("--roi_y", default=48, type=int, help="roi size in y direction")
p.add_argument("--roi_z", default=48, type=int, help="roi size in z direction")

p.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
p.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")       ## 镜像旋转
p.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability") ## 角度旋转
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
p.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
p.add_argument("--mlp_dim", default=1024, type=int, help="mlp dimention in ViT encoder")
p.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")      ## 多少个TF
p.add_argument("--feature_size", default=8, type=int, help="feature size dimention")
p.add_argument("--in_channels", default=1, type=int, help="number of input channels")
p.add_argument("--out_channels", default=1, type=int, help="number of output channels")

p.add_argument('--epochs',type=int,default=20)

args = p.parse_known_args()[0]

