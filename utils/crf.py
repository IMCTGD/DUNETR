#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   09 January 2019


import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        b, C, l, H, W = probmap.shape

        # probmap = probmap.unsqueeze(0)
        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)
        U = U.reshape((C, -1))
        # print('!@#' * 20)
        # print(probmap.shape)
        # print(l)
        # print(U.shape)

        image = np.ascontiguousarray(image)
        # print(image.shape[2:5])

        d = dcrf.DenseCRF(W*H*l, C)
        d.setUnaryEnergy(U) ## 得到一元势 (负对数概率)
        # 这将创建与颜色无关的功能，然后将它们添加到CRF中
        feats = create_pairwise_gaussian(sdims=(3, 3, 3), shape=image.shape[2:5])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
        # d.addPairwiseEnergy(feats, compat=3,kernel = dcrf.FULL_KERNEL, normalization = dcrf.NORMALIZE_SYMMETRIC)
        # 这将创建与颜色相关的功能，然后将它们添加到CRF中
        # feats = create_pairwise_bilateral(sdims=(80, 80, 80), schan=(13, 13, 13), img=image, chdim=2)
        # d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
        # print(image.shape)

        ## 2D
        # d = dcrf.DenseCRF2D(W, H, C)
        # d.setUnaryEnergy(U)
        # d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        # d.addPairwiseBilateral(
        #     sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        # )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape(probmap.shape[1:5])

        return Q
