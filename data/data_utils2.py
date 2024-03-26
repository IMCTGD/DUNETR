# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader2(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),     ## 加载字典
            transforms.AddChanneld(keys=["image", "label"]),    ## 增加色彩通道，如果有的话就不增肌了
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),    ## 重定向轴方向
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),  ## 图像重采样，
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),  ## 图像值强度变换,CT和MRI的值都是从-1000—+3000多的不等，通常需要进行归一化,选定范围区间和归一后的范围
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),    ## 裁剪前景
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),                      ## ROI的大小
                pos=1,                                                                  ## 阳性值的概率
                neg=1,                                                                  ## 阴性值的概率为
                num_samples=2,                                                          ## 返回子图
                image_key="image",
                image_threshold=0,
            ),   ## 按阴性阳性比裁剪
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),    ## 随机翻转
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),   ## 90度角度翻转
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            # transforms.CenterSpatialCropd(keys=['image', 'label'], roi_size=(96,96,96)), ## 中心裁剪
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.CenterSpatialCropd(keys=['image', 'label'], roi_size=(96, 96, 96)),  ## 中心裁剪
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    if args.test_mode:
        # test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        # print(datalist_json)
        # print(data_dir)
        # print(qw)
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        # print('ahhhhhhhhhhhhhhhhhhhhhhhhh')
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=False,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        # print('--FUCK--')
        # print(data_dir)
        # print(qw)
        ## json文件地址，是否是分割任务，加载哪个数据集，数据集地址
        # print(datalist_json)
        # datalist = load_decathlon_datalist('F:/data/data_1100/dataset_0.json', True, "training",
        #                                    base_dir='F:\\data\\data_1100\\chosen_dia\\10011174\\img.nii.gz')
        # print('!@@##$%'*30)
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            ## 把数据加载储存起来，注意：若数据量大内存小的话，会爆内存
            # train_ds = data.CacheDataset(
            #     data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            # )
            train_ds = data.Dataset(
                data=datalist, transform=train_transform
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )


        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        # print('@!!#')
        # # print(train_loader.__iter__().next())
        # print(qw)
        loader = [train_loader, val_loader]

    return loader

if __name__ == '__main__':
    print('star-------')
    import argparse

    parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
    parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument(
        "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
    )
    parser.add_argument("--data_dir", default="../dataset/dataset0/", type=str, help="dataset directory")
    parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
    parser.add_argument(
        "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
    )
    parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
    parser.add_argument("--max_epochs", default=100, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
    parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--workers", default=1, type=int, help="number of workers")  ## 多线程
    parser.add_argument("--model_name", default="unetr", type=str, help="model name")
    parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
    parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
    parser.add_argument("--res_block", action="store_true", help="use residual blocks")
    parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
    parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float,
                        help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float,
                        help="RandShiftIntensityd aug probability")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
    parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
    parser.add_argument("--resume_jit", action="store_true",
                        help="resume training from pretrained torchscript checkpoint")
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    args = parser.parse_args()
    args.test_mode = False

    # loader = get_loader(args)
    print()