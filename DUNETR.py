#  开发人员：    根深蒂固~
#  开发时间：    22/12/2022 下午7:35
#  功能作用：    未知

import torch
import torch.nn as nn
from typing import Tuple, Union
from monai.networks.nets import ViT
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock


class Fusion_block(nn.Module):
    def __init__(self, hidden_size,feature_size, up1, up2):
        super(Fusion_block, self).__init__()

        ## 把transformer转成conv
        self.tf = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size,
            num_layer=up1,  ## 上采样块的数量,0:放大1次，1：放大2次，2：放大3次
            kernel_size=3,
            stride=1,
            upsample_kernel_size=up2,  ## 转置卷积层的卷积核大小，每次放大多少倍
            norm_name='instance',
            conv_block=False,  ## 是否使用卷积块的 bool 参数
            res_block=True,
        )

        # 把双倍的通道变成1倍
        self.conv = UnetrBasicBlock(
            spatial_dims=3,  ## 空间维数
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',  ## 特征规范化类型和参数: instance、batch
            res_block=True,  ## 否使用残差块的 bool 参数
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f, c):
        _, channl, _, _, _ = c.shape
        f = self.tf(f)
        fuse = torch.cat([f, c], dim=1)
        fuse = self.conv(fuse)
        return fuse


class DUNETR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")
        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12                ## 多少次
        self.patch_size = (16, 16, 16)      ## 采样块
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,  ## 输入通道
            img_size=img_size,  ## 图像大小
            patch_size=self.patch_size,  ## 采样块大小
            hidden_size=hidden_size,  ## 隐藏层线性大小
            mlp_dim=mlp_dim,  ## MLP线性大小
            num_layers=self.num_layers,  ## 多少个VIM
            num_heads=num_heads,  ## 多头
            pos_embed=pos_embed,  ## 编码
            classification=self.classification,  ## 是否分类
            dropout_rate=dropout_rate,
        )

        self.connect1 = UnetrBasicBlock(
            spatial_dims=3,  ## 空间维数
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,  ## 特征规范化类型和参数: instance、batch
            res_block=res_block,  ## 否使用残差块的 bool 参数
        )

        ## 上采样
        self.up1 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 8,
            num_layer=0,  ## 上采样块的数量,0:放大1次，1：放大2次，2：放大3次
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,  ## 转置卷积层的卷积核大小，每次放大多少倍
            norm_name=norm_name,
            conv_block=conv_block,  ## 是否使用卷积块的 bool 参数
            res_block=res_block,
        )
        self.up2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            num_layer=0,  ## 上采样块的数量,0:放大1次，1：放大2次，2：放大3次
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,  ## 转置卷积层的卷积核大小，每次放大多少倍
            norm_name=norm_name,
            conv_block=conv_block,  ## 是否使用卷积块的 bool 参数
            res_block=res_block,
        )
        self.up3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            num_layer=0,  ## 上采样块的数量,0:放大1次，1：放大2次，2：放大3次
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,  ## 转置卷积层的卷积核大小，每次放大多少倍
            norm_name=norm_name,
            conv_block=conv_block,  ## 是否使用卷积块的 bool 参数
            res_block=res_block,
        )
        self.up4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            num_layer=0,  ## 上采样块的数量,0:放大1次，1：放大2次，2：放大3次
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,  ## 转置卷积层的卷积核大小，每次放大多少倍
            norm_name=norm_name,
            conv_block=conv_block,  ## 是否使用卷积块的 bool 参数
            res_block=res_block,
        )

        ## 下采样
        self.down4 = UnetrBasicBlock(
            spatial_dims=3,  ## 空间维数
            in_channels=feature_size,
            out_channels=feature_size * 2,
            kernel_size=5,
            stride=2,
            norm_name=norm_name,  ## 特征规范化类型和参数: instance、batch
            res_block=res_block,  ## 否使用残差块的 bool 参数
        )

        self.down3 = UnetrBasicBlock(
            spatial_dims=3,  ## 空间维数
            in_channels=feature_size * 2,
            out_channels=feature_size * 4,
            kernel_size=5,
            stride=2,
            norm_name=norm_name,  ## 特征规范化类型和参数: instance、batch
            res_block=res_block,  ## 否使用残差块的 bool 参数
        )
        self.down2 = UnetrBasicBlock(
            spatial_dims=3,  ## 空间维数
            in_channels=feature_size * 4,
            out_channels=feature_size * 8,
            kernel_size=5,
            stride=2,
            norm_name=norm_name,  ## 特征规范化类型和参数: instance、batch
            res_block=res_block,  ## 否使用残差块的 bool 参数
        )
        self.down1 = UnetrBasicBlock(
            spatial_dims=3,  ## 空间维数
            in_channels=feature_size * 8,
            out_channels=feature_size * 8,
            kernel_size=5,
            stride=2,
            norm_name=norm_name,  ## 特征规范化类型和参数: instance、batch
            res_block=res_block,  ## 否使用残差块的 bool 参数
        )

        ## 拼接部分
        self.cat1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.cat2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.cat3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.cat4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size ,
            out_channels=feature_size ,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        ## 融合模块
        self.fus2 = Fusion_block(hidden_size, feature_size * 8, 0, 1)
        self.fus3 = Fusion_block(hidden_size, feature_size * 8, 0, 2)
        self.fus4 = Fusion_block(hidden_size, feature_size * 4, 1, 2)
        self.fus5 = Fusion_block(hidden_size, feature_size * 2, 2, 2)

        ## 输出部分
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)

    ## 转成 C,W,H,l模式
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)    ## 4, 6, 6, 6, 768
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):                                                        ## 输入：4, 1, 64, 128, 128
        # print(x_in.shape)
        # print(qwe)
        ## transformer部分
        x, hidden_states_out = self.vit(x_in)                                       ## 1, 256, 768
        u1 = self.connect1(x_in)                                                    ## F, 64, 128, 128

        ## UNet编码过程
        d5 = self.down4(u1)                                                         ## F*2, 32, 64, 64
        d4 = self.down3(d5)                                                         ## F*4, 16, 32, 32
        d3 = self.down2(d4)                                                         ## F*8, 8, 16, 16
        d2 = self.down1(d3)                                                         ## F*8, 4, 8, 8

        ## 底部U型
        u2 = self.proj_feat(x, self.hidden_size, self.feat_size)                     ## 768, 4, 8, 8

        ## 融合部分
        u22 = self.fus2(u2,d2)
        u3 = self.fus3(self.proj_feat(hidden_states_out[9], self.hidden_size, self.feat_size), d3)  ## F*8, 8, 16, 16
        u4 = self.fus4(self.proj_feat(hidden_states_out[6], self.hidden_size, self.feat_size), d4)  ## F*4, 16, 32, 32
        u5 = self.fus5(self.proj_feat(hidden_states_out[3], self.hidden_size, self.feat_size), d5)  ## F*2, 32, 64, 64

        ## 解码过程
        u33 = self.up1(u22)                                                         ## F*8, 8, 16, 16
        u44 = self.up2(self.cat1(u33,u3))                                           ## F*4 , 16, 32, 32
        u55 = self.up3(self.cat2(u44,u4))                                           ## F*2 , 32, 64, 64
        u11 = self.up4(self.cat3(u55,u5))                                           ## F, 8, 64, 128, 128
        out = self.cat4(u11,u1)                                                     ## F, 64, 128, 128

        logits = self.out(out)                                                      ## 4, output, 96, 96, 96


        # # a = self.cat1(u33,u3)
        # print('*!'*20)
        # print(logits.shape)
        # print(qw)
        return logits
