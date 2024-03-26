import numpy as np
import nibabel as nib
from skimage.measure import label
import math
import dgl
import torch as th
from dgl.data.utils import save_graphs, load_graphs
import os
import re
from utils.Calculate_metrics import get_region_num
from utils.utils import dijkstra

Inf = math.inf

def convert_np_tree(img):
    '''
    :param img:centerline img,sure this img is connection
    :return: tree_struct_data
    '''
    # find the points of centerline
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
    adj = adj + adj.T
    adj[adj > 1.8] = 0

    # find leaf
    leaf = []
    leaf_loc = []
    for i in range(point_nums):
        p = adj[i, :]
        if np.sum(p != 0) == 1:
            leaf.append(i)
            leaf_loc.append(loc[i, :])

    adj[adj == 0] = float('inf')
    for i in range(point_nums):
        adj[i, i] = 0

    if len(leaf_loc)<=1:

        return None,None

    leaf_loc = np.array(leaf_loc)
    leaf = np.array(leaf)
    z = leaf_loc[:, 2]
    root = np.argmax(z)
    root_index = leaf[root]
    # root_node = leaf_loc[root, :]
    path_dict = dict()
    node_dict = dict()
    node_dict['root'] = root_index
    node_dict['loc'] = loc
    node_dict['leaves'] = leaf

    # path_dict['leaf_node']
    for i in leaf:
        if i != root_index:
            p = dijkstra(adj.copy(), root_index, i)
            path_dict[str(i)] = p
    return path_dict, node_dict

def get_picture2d(node_loc, img, label, patch_size):
    # 获取纵轴的切片
    x_start, x_end = node_loc[0] - patch_size // 2, node_loc[0] + patch_size // 2
    y_start, y_end = node_loc[1] - patch_size // 2, node_loc[1] + patch_size // 2

    if x_start < 0:
        x_start = 0
        x_end = x_start + patch_size
    if x_end > img.shape[0]:
        x_end = img.shape[0]
        x_start = x_end - patch_size

    if y_start < 0:
        y_start = 0
        y_end = y_start + patch_size
    if y_end > img.shape[1]:
        y_end = img.shape[1]
        y_start = y_end - patch_size

    img_2d = img[x_start:x_end, y_start:y_end, node_loc[2]]
    label_2d = label[x_start:x_end, y_start:y_end, node_loc[2]]
    i, j = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end), indexing='ij')
    k=np.zeros_like(i,dtype=np.int)+node_loc[2]

    return np.array([img_2d, label_2d, i, j,k])


def get_picture3d(node_loc, img, label, patch_size):
    # 获取纵轴的切片
    x_start, x_end = node_loc[0] - patch_size[0] // 2, node_loc[0] + patch_size[0] // 2
    y_start, y_end = node_loc[1] - patch_size[1] // 2, node_loc[1] + patch_size[1] // 2
    z_start, z_end = node_loc[2] - patch_size[2], node_loc[2] + patch_size[2]

    if x_start < 0:
        x_start = 0
        x_end = x_start + patch_size[0]
    if x_end > img.shape[0]:
        x_end = img.shape[0]
        x_start = x_end - patch_size[0]

    if y_start < 0:
        y_start = 0
        y_end = y_start + patch_size[1]
    if y_end > img.shape[1]:
        y_end = img.shape[1]
        y_start = y_end - patch_size[1]

    z_start = 0 if z_start < 0 else z_start
    z_end=z_start+patch_size[-1]

    z_end = img.shape[2] if z_end > img.shape[2] else z_end
    z_start=z_end-patch_size[-1]

    img_3d = img[x_start:x_end, y_start:y_end, z_start:z_end]
    label_3d = label[x_start:x_end, y_start:y_end, z_start:z_end]


    i, j, k = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end), np.arange(z_start, z_end),
                          indexing='ij')
    # print('i',i.shape)
    # print('label_2d',label_2d.shape)
    # k=np.zeros_like(i,dtype=np.int)+node_loc[2]


    return np.array([img_3d, label_3d, i, j, k])


def convert_dgl(path_set, img_list, path_dict):
    # 根据path_set编号
    path_set = path_set.tolist()
    g = dgl.DGLGraph()
    g.add_nodes(len(path_set))
    for k in path_dict.keys():
        p = path_dict[k]
        for index in range(len(p) - 1):
            dgl_src = path_set.index(p[index + 1])
            dgl_dst = path_set.index(p[index])
            if g.has_edges_between(dgl_src, dgl_dst):
                continue
            g.add_edges(dgl_src, dgl_dst)
            g.nodes[dgl_src].data['data'], g.nodes[dgl_src].data['label'] = th.tensor(
                np.expand_dims(img_list[dgl_src, 0, :, :, :], axis=0)), th.tensor(
                np.expand_dims(img_list[dgl_src, 1, :, :, :], axis=0))
            g.nodes[dgl_src].data['loc'] = th.tensor(np.expand_dims(img_list[dgl_src, 2:5, :, :, :], axis=0))

            g.nodes[dgl_dst].data['data'], g.nodes[dgl_dst].data['label'] = th.tensor(
                np.expand_dims(img_list[dgl_dst, 0, :, :, :], axis=0)), th.tensor(
                np.expand_dims(img_list[dgl_dst, 1, :, :, :], axis=0))
            g.nodes[dgl_dst].data['loc'] = th.tensor(np.expand_dims(img_list[dgl_dst, 2:5, :, :, :], axis=0))

    return g


def recover_img(g, img):
    # img=img_nii.get_fdata()
    loc = g.ndata['loc'].numpy()
    pre_label = g.ndata['pre_label'].numpy()
    pre_label = np.squeeze(pre_label, axis=1)
    X, Y, Z = loc[:, 0, :, :], loc[:, 1, :, :], loc[:, 2, :, :]
    # X=np.ascontiguousarray(X).flatten().astype(np.int)
    # Y = np.ascontiguousarray(Y).flatten().astype(np.int)
    # Z = np.ascontiguousarray(Z).flatten().astype(np.int)
    # pre_label=np.ascontiguousarray(pre_label).flatten()
    X = X.astype(np.int)
    Y = Y.astype(np.int)
    Z = Z.astype(np.int)

    img_recover = np.zeros_like(img)
    img_recover[X, Y, Z] = pre_label

    return img_recover


def recover_img_run(id, save_path, o_path, graph_path):
    img_nii = nib.load(os.path.join(o_path, id, 'img.nii.gz'))
    img = img_nii.get_fdata()
    img_new = np.zeros_like(img)
    for i in os.listdir(graph_path):
        if re.search(id, i):
            g = load_graphs(os.path.join(graph_path, i), [0])[0][0]
            img_new = img_new + recover_img(g, img)
    img_new[img_new > 0.5] = 1
    img_new[img_new <= 0.5] = 0
    os.makedirs(os.path.join(save_path, id), exist_ok=True)
    nib.save(nib.Nifti1Image(img_new, img_nii.affine), os.path.join(save_path, id, 'pre_32.nii.gz'))


class Convert_tree:
    def __init__(self, cl_path, img_path, label_path, save_graph_path, data_type, patch_size):
        '''
        :param cl_path: pre centerline image
        :param img_path: image
        :param label_path: label
        :param save_graph_path: save graph path
        :param data_type: train or valid
        :param patch_size: len(patch_size)==3 [x,y,z]
        '''

        self.cl_path = cl_path
        self.img_path = img_path
        self.label_path = label_path
        self.save_graph_path = save_graph_path
        self.data_type = data_type
        self.patch_size = patch_size

    def run_convert(self, id):
        print(id)
        cl_path = os.path.join(self.cl_path, id, 'pre_cl.nii.gz')
        img_path = os.path.join(self.img_path, id, 'img.nii.gz')
        label_path = os.path.join(self.label_path, id, 'label.nii.gz')

        all_cl = nib.load(cl_path).get_fdata()
        img = nib.load(img_path).get_fdata()
        true_label = nib.load(label_path).get_fdata()
        a = get_region_num(all_cl, 1)
        b = get_region_num(all_cl, 2) - a
        ab_list = [a,b]

        dt = self.data_type

        if len(self.patch_size)==3:
            g_picture=get_picture3d
        else:
            g_picture=get_picture2d

        for index, ii in enumerate(ab_list):
            if ii.sum() <=5 :
                continue
            cl = ii
            path_dict, node_dict = convert_np_tree(cl)

            if path_dict==None:
                continue

            path_list = []
            for k in path_dict.keys():
                path_list = path_list + path_dict[k]
            path_set = np.unique(np.array(path_list))
            img_list = []
            for i in range(len(path_set)):
                node = node_dict['loc'][path_set[i], :]
                image3d = g_picture(node, img, true_label, self.patch_size)
                img_list.append(image3d)
            img_list = np.array(img_list)
            g = convert_dgl(path_set, img_list, path_dict)
            os.makedirs(os.path.join(self.save_graph_path, 'patch_%d_%d_%d' % (self.patch_size[0],self.patch_size[1],
                                                                               self.patch_size[2]), dt), exist_ok=True)
            if g.num_nodes()!=0:
                save_graphs(
                    os.path.join(self.save_graph_path, 'patch_%d_%d_%d' % (self.patch_size[0],self.patch_size[1],
                                                                           self.patch_size[2]), dt, '%s_g%d.bin' % (id, index)),
                    [g])

class Recover_img:
    def __init__(self,img_path,save_path,graph_path,save_file_name='pre_32.nii.gz'):
        self.graph_path=graph_path
        self.img_path=img_path
        self.save_path=save_path
        self.save_file_name=save_file_name

    def recover_img_run(self,id):
        print(id)
        img_nii = nib.load(os.path.join(self.img_path, id, 'img.nii.gz'))
        img = img_nii.get_fdata()
        img_new = np.zeros_like(img)
        for i in os.listdir(self.graph_path):
            if re.search(id, i):
                g = load_graphs(os.path.join(self.graph_path, i), [0])[0][0]
                img_new = img_new + recover_img(g, img)
        img_new[img_new > 0.5] = 1
        img_new[img_new <= 0.5] = 0
        os.makedirs(os.path.join(self.save_path, id), exist_ok=True)
        nib.save(nib.Nifti1Image(img_new, img_nii.affine), os.path.join(self.save_path, id, self.save_file_name))



















