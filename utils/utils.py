import pandas as pd
import numpy as np
import math
import scipy.linalg as linalg
from scipy.ndimage.interpolation import zoom
from monai.transforms import RandFlip, RandRotate


def get_csv_split(csv_file, k):
    '''
    :param csv_file: 划分表
    :param k: 第几折
    :return:
    '''
    id_pd = pd.read_csv(csv_file, dtype=str)
    ID_split = dict()
    ID_split['train'] = id_pd[id_pd['k_fold'] != str(k)]['ID'].to_list()
    ID_split['valid'] = id_pd[id_pd['k_fold'] == str(k)]['ID'].to_list()
    return ID_split


def get_line(x_start, x_end, dim=3):
    v = x_end - x_start
    d = np.sqrt(np.sum(v ** 2))
    if d == 0:
        return x_start.reshape(1, 3)
    v = v / d
    d_len = np.floor(d)
    d_stride = np.arange(d_len)
    d_stride = d_stride[:, np.newaxis]
    d_stride = d_stride.repeat(dim, axis=1)
    x_point = x_start[np.newaxis, :] + d_stride * v
    return x_point

# 旋转矩阵 欧拉角
def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix


def loc_convert(loc, axis, radian):
    '''

    :param loc:原始坐标
    :param axis: 绕旋转的轴（点）
    :param radian: 弧度
    :return: 新坐标
    '''
    radian = np.deg2rad(radian)
    new_loc = np.dot(rotate_mat(axis, radian), loc)
    return new_loc


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

def Random_reshape_img(img, output_shape, order=0):
    '''
    :param img: 3D array
    :param output_shape: 输出图像大小
    :param order: 插值方式
    :return: 插值后的图像
    '''
    s = (output_shape[0] / img.shape[0], output_shape[1] / img.shape[1], 1)    ## 不改变Z轴
    new_img = zoom(img, zoom=s, order=order)

    return new_img


def normalize(img):
    img = img / 4000
    img[img > 1] = 1
    img[img < -1] = -1
    return img


def dijkstra(mat, begin, end):
    x = float('inf')
    n = len(mat)
    parent = []  # 用于记录每个结点的父辈结点
    collected = []  # 用于记录是否经过该结点
    distTo = mat[begin]  # 用于记录该点到begin结点路径长度,初始值存储所有点到起始点距离
    path = []  # 用于记录路径
    for i in range(0, n):  # 初始化工作
        if i == begin:
            collected.append(True)  # 所有结点均未被收集
        else:
            collected.append(False)
        parent.append(-1)  # 均不存在父辈结点
    while True:
        if collected[end] == True:
            break
        min_n = x
        for i in range(0, n):
            if collected[i] == False:
                if distTo[i] < min_n:  # 代表头结点
                    min_n = distTo[i]
                    v = i
        collected[v] = True
        for i in range(0, n):
            if (collected[i] == False) and (distTo[v] + mat[v][i] < distTo[i]):  # 更新最短距离
                parent[i] = v
                distTo[i] = distTo[v] + mat[v][i]
    e = end
    while e != -1:  # 利用parent-v继承关系，循环回溯更新path并输出
        path.append(e)
        e = parent[e]
    path.append(begin)
    path.reverse()
    return path


def extract_slice(img, c, v, radius):
    '''
    :param V:3d 图像
    :param center: 中心（N,3）
    :param normal: 法向量（N,3）
    :param radius: 边长
    :return:
    slicer：得到的2d切片
    loc: 得到切片对应的原3d坐标
    '''
    N = v.shape[0]

    epsilon = 1e-12
    x = np.arange(-radius, radius, 1)
    y = np.arange(-radius, radius, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    loc = np.array([X.flatten(), Y.flatten(), Z.flatten()])

    hspInitialVector = np.array([0, 0, 1])
    h_norm = np.linalg.norm(hspInitialVector)
    h_v = hspInitialVector / h_norm
    h_v[h_v == 0] = epsilon

    v = v / np.linalg.norm(v, axis=1).reshape(N, 1)
    v[v == 0] = epsilon

    loc = loc_convert(loc, [0, 0, 1], 0)

    hspVecXvec = np.cross(h_v, v) / np.linalg.norm(np.cross(h_v, v), axis=1).reshape(v.shape[0], 1)
    h_v = h_v[np.newaxis, :]
    acosineVal = np.arccos(np.dot(v, h_v.T))
    hspVecXvec[np.isnan(hspVecXvec)] = epsilon
    acosineVal[np.isnan(acosineVal)] = epsilon

    loc_arr = []
    for i in range(v.shape[0]):
        loc_arr.append(loc_convert(loc, hspVecXvec[i, :], 180 * acosineVal[i, :] / math.pi))
    loc_arr = np.array(loc_arr)

    sub_loc = loc_arr + c[:, :, np.newaxis]
    loc = np.round(sub_loc)

    loc = np.reshape(loc, (N, 3, X.shape[0], X.shape[1]))  # (N,3,64,64)
    sliceInd = np.zeros((N, X.shape[0], X.shape[1]))
    sliceInd[sliceInd == 0] = np.nan
    slicer = np.copy(sliceInd)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            ll = loc[:, :, i, j]
            flag = (0 <= ll[:, 0]) & (ll[:, 0] < img.shape[0]) & (0 <= ll[:, 1]) & (ll[:, 1] < img.shape[1]) & (
                    0 <= ll[:, 2]) & (ll[:, 2] < img.shape[2])
            slicer[flag, i, j] = img[ll[flag, 0].astype(np.int), ll[flag, 1].astype(np.int), ll[flag, 2].astype(np.int)]

    loc = np.transpose(loc, axes=(0, 2, 3, 1))  # (N,64,64,3)
    loc[loc[:, :, :, 0] >= img.shape[0], 0] = 0
    loc[loc[:, :, :, 1] >= img.shape[1], 1] = 0
    loc[loc[:, :, :, 2] >= img.shape[2], 2] = 0
    loc = np.transpose(loc, axes=(0, 3, 1, 2))  # (N,64,64,3)

    slicer[np.isnan(slicer)] = 0

    loc[loc < 0] = 0
    loc = loc.astype(np.int)
    return slicer, sub_loc, loc

def convert_np_graph(img):
    '''
    :param 中心线的图像
    :return: 三个字典（1）路径字典path_dict,其中路径的字典的键为叶子节点的编号，存储了根节点到该叶子节点的路径
    （2）节点字典node_dict，其中有三个键['root','leaf','loc'],其中root对应的值根节点的编号，leaf对应的叶子节点的编号为列表，loc对应了节点在原图像上的空间坐标
    （3）以及节点对应的法向量字典normal_dict，键为节点的编号，值为节点对应的法向量
    '''
    # 冠状动脉分段

    # 找到中心线点的位置
    x, y, z = np.where(img == 1)
    point_nums = len(x)
    loc = np.array([x, y, z]).T

    # generate matrix of adj
    adj = np.zeros((point_nums, point_nums))

    # calculate adj
    for i in range(point_nums):
        for j in range(i + 1, point_nums):
            d = np.sqrt(np.sum((loc[i, :] - loc[j, :]) ** 2))
            adj[i, j] = d
    #

    adj = adj + adj.T
    adj[adj > np.sqrt(3)] = 0

    # find leaf
    leaf = []
    leaf_loc = []
    for i in range(point_nums):
        p = adj[i, :]
        if np.sum(p != 0) == 1:
            leaf.append(i)
            leaf_loc.append(loc[i, :])


    # adj[adj == 0] = float('inf')
    for i in range(point_nums):
        adj[i, i] = 0

    leaf_loc = np.array(leaf_loc)
    leaf = np.array(leaf)

    z = leaf_loc[:, 2]

    # 将垂直方向上最大值点的作为主动脉
    root = np.argmax(z)
    root_index = leaf[root]

    path_dict = dict()
    node_dict = dict()
    normal_dict = dict()
    node_dict['root'] = root_index
    node_dict['loc'] = loc
    node_dict['leaves'] = leaf


    # path_dict['leaf_node']
    path_list=[]
    index=root_index
    path_list.append(root_index)
    for i in range(point_nums):
        neighbor=np.where(adj[index,:]!=0)[0]
        # print(np.sum(adj[index,:]!=0))
        # print(index)
        for n in neighbor:
            if n not in path_list:
                # print(n)
                path_list.append(n)
                index=n
                continue
    path_dict[str(path_list[-1])]=path_list
    p_normal = np.gradient(loc[path_list, :], axis=0)
    node_dict['normal']=p_normal
    for index,i in enumerate(path_list):
        normal_dict[str(i)] = p_normal[index, :]

    return path_dict, node_dict, normal_dict


class Transform:
    def __init__(self, flip=0.5, rotate=0.5):
        self.flip = flip
        self.rotate = rotate

    def transform(self, img):
        tran1 = RandFlip(prob=self.flip)
        img = tran1(img)
        tran2 = RandRotate(prob=self.rotate)
        img = tran2(img)
        return img
