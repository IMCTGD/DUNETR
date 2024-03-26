from torch.utils.data import Dataset, DataLoader
from dgl.data.utils import load_graphs
import os
import collections
import dgl
import torch
import yaml
import argparse
from utils.utils import get_csv_split


class Tree_Batch(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.trees = os.listdir(data_dir)

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        tree_index = self.trees[index]
        tree_path = os.path.join(self.data_dir, tree_index)
        g = load_graphs(tree_path, [0])[0][0]
        g.ndata['data']=torch.unsqueeze(g.ndata['data'],dim=1).float()
        g.ndata['label'] = torch.unsqueeze(g.ndata['label'], dim=1).float()
        # print(g.ndata['data'])
        # sampler={'id_index':tree_index,'g':g}
        return g


class Tree_inference(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.trees = os.listdir(data_dir)

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        tree_index = self.trees[index]
        tree_path = os.path.join(self.data_dir, tree_index)
        g = load_graphs(tree_path, [0])[0][0]
        g.ndata['data']=torch.unsqueeze(g.ndata['data'],dim=1).float()
        g.ndata['label'] = torch.unsqueeze(g.ndata['label'], dim=1).float()
        s={'id_index':tree_index,'g':g}
        return s


Train_Batch = collections.namedtuple('Train_Batch', ['graph','data','label'])
def train_batcher(device):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return Train_Batch(graph=batch_trees.to(device),
                        data=batch_trees.ndata['data'].to(device).float(),
                        label=batch_trees.ndata['label'].to(device).float())
    return batcher_dev

Valid_Batch = collections.namedtuple('Valid_Batch', ['graph','data','label'])
def valid_batcher(device):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return Valid_Batch(graph=batch_trees.to(device),
                        data=batch_trees.ndata['data'].to(device).float(),
                        label=batch_trees.ndata['label'].to(device).float())
    return batcher_dev




def test_load():
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--Direct_model', type=str, default='FCN')
    p.add_argument('--Direct_parameter', type=str, default='Mid_resolution_4_Dice')
    p.add_argument('--pools', type=int, default=4)
    p.add_argument('--patch_size', type=int, default=32)
    p.add_argument('--z_size', type=int, default=4)



    args = p.parse_args()
    config_file = args.config_file
    k = args.fold
    load_num = args.load_num
    b_size = args.batch_size
    version = args.version_name
    p_size = args.p_size
    gpu_index = args.gpu_index
    mode = args.mode

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(r'config/config.yaml') as f:
        config = yaml.load(f)
        # patch_type=config['General_parameters'][overlap]
        epochs = config['General_parameters']['epoch']
        csv_path = config['General_parameters']['csv_path']
        data_path = config['General_parameters']['data_path']
        patch_path = config['General_parameters']['patch_path']
        patch_size = config['General_parameters']['patch_size']

    id_dict = get_csv_split(csv_path, 1)

    # 设置结果路径
    save_graph_path = 'result/%s/fold_%d/patches/patch_%d_%d_%d/save_graph' % (
    version, k, patch_size[0], patch_size[1], patch_size[2])
    save_model_path = 'result/%s/fold_%d/patches/patch_%d_%d_%d/model_save' % (
    version, k, patch_size[0], patch_size[1], patch_size[2])
    recover_path = 'result/%s/fold_%d/patches/patch_%d_%d_%d/pre_label' % (
    version, k, patch_size[0], patch_size[1], patch_size[2])
    os.makedirs(save_graph_path, exist_ok=True)
    os.makedirs(save_model_path, exist_ok=True)
    os.makedirs(recover_path, exist_ok=True)

    # 加载数据
    train_path = os.path.join(patch_path, 'fold_%d' % k, 'patch_%d_3' % patch_size[0], 'train')
    valid_path = os.path.join(patch_path, 'fold_%d' % k, 'patch_%d_3' % patch_size[0], 'valid')
    train_data = Tree_inference(train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=b_size, shuffle=True, collate_fn=train_batcher(device),
                              num_workers=0)

    valid_data = Tree_Batch(valid_path)
    valid_loader = DataLoader(dataset=valid_data, batch_size=b_size, shuffle=False, collate_fn=valid_batcher(device),
                              num_workers=0)

    for index, i in enumerate(train_data):
        a = i
