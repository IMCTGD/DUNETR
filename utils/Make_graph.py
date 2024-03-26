import numpy as np
from skimage.filters import sobel
import nibabel as nib
from utils.utils import loc_convert, dijkstra, get_line, extract_slice
from utils.Calculate_metrics import get_region_num
from utils.utils import convert_np_graph
import dgl
from scipy.ndimage.interpolation import zoom
from skimage.morphology import skeletonize
from skimage.measure import label as sl, marching_cubes
import torch as th
from dgl.data.utils import save_graphs, load_graphs
import os
from scipy import ndimage
import glob


def get_contour(img):
    contour = sobel(img)
    # contour[contour!=0]=1
    contour[contour > 0.6] = 1
    contour[contour <= 0.6] = 0
    return contour


def get_point_feature(point, normal, image, contour, spacing, radiu_stride=32, tangle=15):
    '''
    该函数输入一个点的坐标，切线方向，以及原图像，对应的轮廓图像以及空间的分辨率,最后得到24个方向的特征+半径
    :param point:中心线点 (N,3)
    :param normal:该中心点上的法向量 (N,3)
    :param image: 3D图像
    :param spacing:体素空间大小 (1,3)
    :return:X 为每一个中心线点的特征 (N,24,33)
    '''
    # 首先得到初始点序列
    # 对于任意过点point并与normal垂直的直线都位于垂直平面上
    # 令x=point[0]-1,y=point[1]-1
    # normal需要经过去零处理
    N = point.shape[0]
    docker = np.zeros_like(image)
    epsilon = 1e-12
    normal[normal == 0] = epsilon
    normal = normal / np.linalg.norm(normal, axis=1).reshape(N, 1)

    # 两次旋转
    # 第一次现在xy平面初始化24射线,初始坐标方向为垂直方向，初始化点为原点
    init_vec = np.array([0, 0, 1]).reshape(1, 3)
    init_point = np.array([0, 0, 0]).reshape(1, 3)
    # 初始化射线方向为（1，0，0）
    radius_vector = np.array([1, 0, 0]).reshape(1, 3)

    # 建立第一条射线上的点
    stride = np.arange(0, radiu_stride + 1)
    stride = stride[:, np.newaxis]
    stride = np.repeat(stride, 3, axis=1)
    init_point_arr = init_point + stride * 0.1 * radius_vector
    point_list = []

    slicer, _, true_loc = extract_slice(docker, point, normal, 32)

    point1_list = []
    for index, i in enumerate(range(0, 360, tangle)):
        point1 = loc_convert(init_point_arr.T, init_vec, i).T + 16
        point1_list.append(point1)

    point1_list = np.array(point1_list)  # (24,33,3)
    point1_list = np.floor(point1_list / spacing).astype(np.int)  # (24,33,3)
    point2 = true_loc[:, :, point1_list[:, :, 0], point1_list[:, :, 1]]  # (N,3,24,33)

    debug = False
    if debug == True:
        a = np.zeros_like(image)
        for p, i in enumerate(point_list):
            index = i
            a[index[-1, 0], index[-1, 1], index[-1, 2]] = p
        print('debug')
        slicer, _, _ = extract_slice(a, point, normal, 32)
        x, y = np.where(slicer == 1)
        # print(print(x.shape))
        # plt.imshow(slicer)
        # plt.show()

    # point_list = np.array(point2)
    point2 = np.transpose(point2, (0, 2, 3, 1))  # (N,24,33,3)
    point2 = point2 * spacing  # (N,24,33,3)

    # 特征和标签，前33维是特征，最后一维是半径长度,最后三维为方向信息
    X = np.zeros((N, 360 // tangle, radiu_stride + 2))

    for i in range(point2.shape[1]):  # for i in 24
        xyz_index = np.round(point2[:, i, :, :] / spacing.reshape(1, 3)).astype(np.int)  # (N,33,3)
        # xyz_index=point_list[i,:,:]
        c_p = np.zeros((xyz_index.shape[0], xyz_index.shape[1]))  # (N,33)
        for p_index in range(xyz_index.shape[1]):  # for p_index in 33

            mid_loc = xyz_index[:, p_index, :]  # (N,3)
            flag = (mid_loc[:, 0] >= 0) & (mid_loc[:, 0] < image.shape[0]) & (mid_loc[:, 1] >= 0) & (
                        mid_loc[:, 1] < image.shape[1]) & (mid_loc[:, 2] >= 0) & (mid_loc[:, 2] < image.shape[2])
            X[flag, i, p_index] = image[mid_loc[flag, 0], mid_loc[flag, 1], mid_loc[flag, 2]]
            c_p[flag, p_index] = contour[mid_loc[flag, 0], mid_loc[flag, 1], mid_loc[flag, 2]]

        edge_index = []
        for n in range(N):
            c_index = np.where(c_p[n, :] != 0)
            if c_index[0].shape[0] == 0:
                contour_index = radiu_stride
            elif c_index[0].shape[0] == 1:
                contour_index = c_index[0][0]
            else:
                contour_index = np.max(c_index[0])
            edge_index.append(contour_index)  # (N)

        edge_point = xyz_index[list(range(xyz_index.shape[0])), edge_index]  # (N,3)

        d = np.sqrt(np.sum((edge_point * spacing - point * spacing) ** 2, axis=1))  # (N,)

        X[:, i, radiu_stride + 1] = d

    dist_map = np.sqrt(np.sum((point2 - point[:, np.newaxis, np.newaxis, :] * spacing) ** 2, axis=3))  # (N,24,33)

    return [X, point2, dist_map]


def divide_cl(cl):
    # cl中只有一个连通域
    # 寻找分支节点
    x, y, z = np.where(cl == 1)
    new_cl = cl.copy()
    point_nums = len(x)
    loc = np.array([x, y, z]).T

    # generate matrix of adj
    adj = np.zeros((point_nums, point_nums))

    # calculate adj
    for i in range(point_nums):
        for j in range(i + 1, point_nums):
            d = np.sqrt(np.sum((loc[i, :] - loc[j, :]) ** 2))
            adj[i, j] = d
    adj = adj + adj.T
    adj[adj > 1.8] = 0

    # find leaf
    for i in range(point_nums):
        p = adj[i, :]
        if np.sum(p != 0) > 2:
            cl[loc[i, 0], loc[i, 1], loc[i, 2]] = 0

    # 统计分段
    cl_rm_branch = sl(cl)

    return cl_rm_branch


def img_resample(img_nii, new_spacing):
    # 读取nii里面的数据
    header = img_nii.header
    spacing = header.get_zooms()
    spacing = np.array(spacing)
    img = img_nii.get_fdata()
    img_shape = np.array(img.shape)
    new_spacing = np.array(new_spacing)

    # 计算新的图片大小
    out_shape = spacing * img_shape / new_spacing
    out_shape = np.round(out_shape).astype(np.int)
    out_shape = out_shape.flatten()
    # 插值

    s = (out_shape[0] / img_shape[0], out_shape[1] / img_shape[1], out_shape[2] / img_shape[2])

    new_image = zoom(img, zoom=s, order=0)

    new_nii = nib.Nifti1Image(new_image, img_nii.affine)
    new_nii.header['pixdim'][1:4] = new_spacing
    return new_nii, new_image


class Construct_graph:
    def __init__(self, cl_all, image, contour, spacing, tangle, radiu_stride=32):
        self.cl_all = cl_all
        self.image = image
        self.contour = contour
        self.spacing = spacing
        self.tangle = tangle
        self.radiu_stride = radiu_stride

    def construct_graph(self, cl):
        '''
        构造图节点以及对应的连接关系，输入为一段冠状动脉的没有分支
        :param cl: 一段冠状动脉的3D中心线图像
        :param image:原始图像
        :param contour:根据标签得到的label图像
        :param spacing:体素空间的大小
        :return:
        '''
        cl_all = self.cl_all
        image = self.image
        contour = self.contour
        spacing = self.spacing
        tangle = self.tangle
        radiu_stride = self.radiu_stride
        # cl = (cl_all == ii).astype(np.int)
        # if cl.sum()<2:
        #     return None
        path_dict, node_dict, normal_dict = convert_np_graph(cl)
        keys = list(path_dict.keys())
        node_path = path_dict[keys[0]]

        loc = node_dict['loc']
        radius_nums = 360 // tangle

        # 初始化图，根据path的位置进行编码
        node_num = loc.shape[0] * radius_nums
        g = dgl.DGLGraph()
        g.add_nodes(node_num)

        # 构图1
        # 初始化
        g.ndata['xv'] = th.zeros(node_num, 33)
        g.ndata['rv'] = th.zeros(node_num)
        g.ndata['dist_map'] = th.zeros(node_num, 33)
        g.ndata['loc'] = th.zeros(node_num, 33, 3)
        g.ndata['layer'] = th.zeros(node_num, 1)
        g.ndata['normal'] = th.zeros(node_num, 33, 3)
        g.ndata['point'] = th.zeros(node_num, 33, 3)
        g.add_edges(th.arange(0, node_num - 1), th.arange(1, node_num))
        g.add_edges(th.arange(1, node_num), th.arange(0, node_num - 1))

        # 构造特征
        point = loc
        normal = node_dict['normal']

        point = point[node_path, :]
        normal = normal[node_path, :]
        feature_list = get_point_feature(point, normal, image, contour, spacing, radiu_stride, tangle)

        X_all = feature_list[0]  # (N,24,33)
        X_loc_all = feature_list[1]  # (N,24,33,3)
        dist_map_all = feature_list[2]  # (N,24,33)

        for i in range(loc.shape[0]):

            X = X_all[i, :, :]
            X_loc = X_loc_all[i, :, :, :]
            dist_map = dist_map_all[i, :, :]

            # print(X)
            # print(dist_map)

            g.ndata['xv'][(i * radius_nums):((i + 1) * radius_nums)] = th.tensor(X[:, :(radiu_stride + 1)])
            g.ndata['rv'][(i * radius_nums):((i + 1) * radius_nums)] = th.tensor(X[:, -1])
            g.ndata['dist_map'][(i * radius_nums):((i + 1) * radius_nums)] = th.tensor(dist_map)
            g.ndata['loc'][(i * radius_nums):((i + 1) * radius_nums)] = th.tensor(X_loc)
            g.ndata['layer'][(i * radius_nums):((i + 1) * radius_nums)] = th.zeros(1, 1) + i
            g.ndata['normal'][(i * radius_nums):((i + 1) * radius_nums)] = th.tensor(normal[i, :])
            g.ndata['point'][(i * radius_nums):((i + 1) * radius_nums)] = th.tensor(point[i, :])

            # 连边
            g.add_edges(i * radius_nums, (i + 1) * radius_nums - 1)
            g.add_edges((i + 1) * radius_nums - 1, i * radius_nums)
            if i != loc.shape[0] - 1:
                g.add_edges(th.arange((i * radius_nums), ((i + 1) * radius_nums)),
                            th.arange(((i + 1) * radius_nums), ((i + 2) * radius_nums)))
                g.add_edges(th.arange(((i + 1) * radius_nums), ((i + 2) * radius_nums)),
                            th.arange((i * radius_nums), ((i + 1) * radius_nums)))

                g.add_edges(th.arange((i * radius_nums), ((i + 1) * radius_nums) - 1),
                            th.arange(((i + 1) * radius_nums) + 1, ((i + 2) * radius_nums)))
                g.add_edges(th.arange(((i + 1) * radius_nums) + 1, ((i + 2) * radius_nums)),
                            th.arange((i * radius_nums), ((i + 1) * radius_nums) - 1))

                g.add_edges(th.arange((i * radius_nums) + 1, ((i + 1) * radius_nums)),
                            th.arange(((i + 1) * radius_nums), ((i + 2) * radius_nums) - 1))
                g.add_edges(th.arange(((i + 1) * radius_nums), ((i + 2) * radius_nums) - 1),
                            th.arange((i * radius_nums) + 1, ((i + 1) * radius_nums)))
                g.add_edges(i * radius_nums, (i + 2) * radius_nums - 1)
                g.add_edges((i + 2) * radius_nums - 1, i * radius_nums)
        return g


def recover_node(g, image, spacing, str_key):
    '''

    :param g:图，一个图等于一个冠状动脉
    :param image: 图像大小
    :param spacing: 体素空间大小
    :return: 3d numpy array
    '''

    label = np.zeros_like(image)
    # label_index = np.zeros_like(image)
    cl = np.zeros_like(image)
    # 半径长度
    rv = g.ndata[str_key].numpy()
    rv = rv.reshape(rv.shape[0], 1)
    # 位置
    loc = np.round(g.ndata['loc'].numpy() / spacing).astype(np.int)
    # 距离
    dist_map = g.ndata['dist_map'].numpy()
    mid_var = np.abs(dist_map - rv)
    min_index = np.argmin(mid_var, axis=1)
    edge_loc = loc[np.arange(dist_map.shape[0]), min_index, :]

    # 连线
    adj = g.adj().to_dense()
    n = g.number_of_nodes()
    for ii in range(n):
        neighbor_node = th.where(adj[ii, :] == 1)[0]
        for j in neighbor_node:
            point1 = edge_loc[ii, :]
            point2 = edge_loc[j, :]

            line = get_line(point1, point2, dim=3)
            line1 = np.floor(line).astype(np.int)
            line2 = np.ceil(line).astype(np.int)
            label[line1[:, 0], line1[:, 1], line1[:, 2]] = 1
            label[line2[:, 0], line2[:, 1], line2[:, 2]] = 1

    center = loc[list(range(0, rv.shape[0], 24)), 0, :]
    x_v = np.array([1, 0, 0]).reshape(1, 3)
    x_v = np.repeat(x_v, repeats=center.shape[0], axis=0)
    x_slicer, x_sub_loc, x_true_loc = extract_slice(label, center, x_v, 32)

    y_v = np.array([0, 1, 0]).reshape(1, 3)
    y_v = np.repeat(y_v, repeats=center.shape[0], axis=0)
    y_slicer, y_sub_loc, y_true_loc = extract_slice(label, center, y_v, 32)

    z_v = np.array([0, 0, 1]).reshape(1, 3)
    z_v = np.repeat(z_v, repeats=center.shape[0], axis=0)
    z_slicer, z_sub_loc, z_true_loc = extract_slice(label, center, z_v, 32)

    for i_index in range(center.shape[0]):
        xx = ndimage.binary_closing(x_slicer[i_index, :, :])
        x_slicer[i_index, :, :] = ndimage.binary_fill_holes(xx)

        yy = ndimage.binary_closing(y_slicer[i_index, :, :])
        y_slicer[i_index, :, :] = ndimage.binary_fill_holes(yy)

        zz = ndimage.binary_closing(z_slicer[i_index, :, :])
        z_slicer[i_index, :, :] = ndimage.binary_fill_holes(zz)

        label[x_true_loc[i_index, 0, :, :], x_true_loc[i_index, 1, :, :], x_true_loc[i_index, 2, :, :]] = \
            label[x_true_loc[i_index, 0, :, :], x_true_loc[i_index, 1, :, :], x_true_loc[i_index, 2, :, :]] + x_slicer[
                                                                                                              i_index,
                                                                                                              :, :]

        label[y_true_loc[i_index, 0, :, :], y_true_loc[i_index, 1, :, :], y_true_loc[i_index, 2, :, :]] = \
            label[y_true_loc[i_index, 0, :, :], y_true_loc[i_index, 1, :, :], y_true_loc[i_index, 2, :, :]] + y_slicer[
                                                                                                              i_index,
                                                                                                              :, :]

        label[z_true_loc[i_index, 0, :, :], z_true_loc[i_index, 1, :, :], z_true_loc[i_index, 2, :, :]] = \
            label[z_true_loc[i_index, 0, :, :], z_true_loc[i_index, 1, :, :], z_true_loc[i_index, 2, :, :]] + z_slicer[
                                                                                                              i_index,
                                                                                                              :, :]

    return label, cl


class Recover_label:
    def __init__(self, data_path, graph_path, pre_label_path, spacing, str_key='pre_rv'):
        self.data_path = data_path
        self.graph_path = graph_path
        self.pre_label_path = pre_label_path
        self.spacing = spacing
        self.str_key = str_key

    def run(self, id):
        print(id)
        img_nii = nib.load(os.path.join(self.data_path, id, 'img.nii.gz'))
        # label_nii = nib.load(os.path.join(self.data_path, id, 'label.nii.gz'))

        # image = img_nii.get_fdata()
        _, image = img_resample(img_nii, self.spacing)

        img_new = np.zeros_like(image)
        # print(os.listdir(save_pre_path))
        # file_list = os.listdir(self.graph_path)
        file_list = glob.glob(os.path.join(self.graph_path, id + '*'))
        if len(file_list) == 0:
            print('没有预测label')
            return
        for i in file_list:
            # print(i)
            g = load_graphs(i, [0])[0][0]
            mid1, mid2 = recover_node(g, image, self.spacing, self.str_key)
            img_new = mid1 + img_new

        img_new[img_new > 0.5] = 1
        img_new[img_new <= 0.5] = 0

        # header = img_nii.header
        new_image_nii=nib.Nifti1Image(img_new, img_nii.affine)
        new_image_nii.header['pixdim'][1:4] = self.spacing
        new_image_nii,_= img_resample(new_image_nii, img_nii.header.get_zooms())

        os.makedirs(os.path.join(self.pre_label_path, id), exist_ok=True)
        nib.save(new_image_nii,
                 os.path.join(self.pre_label_path, id, '%s.nii.gz'%self.str_key))
        print('%s:done' % id)


class Make_Graph:
    def __init__(self, image_path, label_path, cl_path, save_graph_path, dtype):
        self.image_path = image_path
        self.label_path = label_path
        self.cl_path = cl_path
        self.save_graph_path = save_graph_path
        self.dtype = dtype

    def run(self, id):
        s_path = os.path.join(self.save_graph_path, self.dtype)
        os.makedirs(s_path, exist_ok=True)
        img_nii = nib.load(os.path.join(self.image_path, id, 'img.nii.gz'))
        label_nii = nib.load(os.path.join(self.label_path, id, 'label.nii.gz'))
        cl_nii = nib.load(os.path.join(self.cl_path, id, 'pre_label.nii.gz'))

        spacing = np.array([0.5, 0.5, 0.5])
        new_image_nii, image = img_resample(img_nii, spacing)
        new_label_nii, label = img_resample(label_nii, spacing)
        new_cl_nii, cl = img_resample(cl_nii, spacing)

        # label = label_nii.get_fdata()
        cl = skeletonize(cl.astype(np.uint8))

        cl = get_region_num(cl, 2)
        cl_branch = divide_cl(cl)
        cl_num = cl_branch.max()

        mk_graph = Construct_graph(cl_branch, image, label, spacing, 15)

        for i in range(1, cl_num + 1):
            per_lc = (cl_branch == i).astype(np.int)
            if per_lc.sum() <= 5:
                continue
            g = mk_graph.construct_graph(per_lc)
            save_graphs(os.path.join(s_path, '%s_%d.bin' % (id, i)), [g])
        print(id + ':成功保存')

    def resample(self, id):
        pass
