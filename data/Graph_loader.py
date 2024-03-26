from torch.utils.data import Dataset, DataLoader
from dgl.data.utils import load_graphs
import os
import collections
import dgl
import torch


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
        g.ndata['data'] = torch.unsqueeze(g.ndata['data'], dim=1).float()
        g.ndata['label'] = torch.unsqueeze(g.ndata['label'], dim=1).float()
        # print(g.ndata['data'])
        # sampler={'id_index':tree_index,'g':g}
        return g


class Graph_loader(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.graph_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, index):
        graph_index = self.graph_list[index]
        tree_path = os.path.join(self.data_dir, graph_index)
        g = load_graphs(tree_path, [0])[0][0]
        # g.ndata['xv'] = torch.unsqueeze(g.ndata['xv'], dim=1).float()
        g.ndata['xv']=normalize(g.ndata['xv'])
        g.ndata['rv'] = torch.unsqueeze(g.ndata['rv'], dim=1).float()
        # print(g.ndata['data'])
        # sampler={'id_index':graph_index,'g':g}
        return g

class Inference_graph(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.graph_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, index):
        graph_index = self.graph_list[index]
        tree_path = os.path.join(self.data_dir, graph_index)
        g = load_graphs(tree_path, [0])[0][0]
        # g.ndata['xv'] = torch.unsqueeze(g.ndata['xv'], dim=1).float()
        g.ndata['xv'] = normalize(g.ndata['xv'])
        g.ndata['rv'] = torch.unsqueeze(g.ndata['rv'], dim=1).float()
        # print(g.ndata['data'])
        sampler={'id_index':graph_index,'g':g}
        return sampler


def normalize(img):
    img = img / 1000
    img[img > 1] = 1
    img[img < -1] = -1
    return img



def collate(samples):
    batched_graph = dgl.batch(samples)
    return batched_graph


if __name__ == '__main__':
    train_path = r'/home/wcb/Python_code/Paper_implementation/Paper_4/graph_data/32_dim/train'
    valid_path = r'/home/wcb/Python_code/Paper_implementation/Paper_4/graph_data/32_dim/valid'

    train_set = Graph_loader(train_path)
    train_loader = DataLoader(train_set, 10, False, collate_fn=collate,num_workers=16)

    for i in train_loader:
        xv=i.ndata['xv'][:,:32]
        print(xv.size())
        print(i.ndata['rv'].size())
