import torch
from torch.utils import data
import numpy as np
from sklearn.utils import murmurhash3_32
import pdb
import pandas as pd
import sklearn.datasets as skds


class SimpleDNADataset(data.Dataset):
    def __init__(self, X_file):
        super(SimpleDNADataset, self).__init__()
        with open(X_file, "r") as f:
            self.d = f.readlines()
        self.length = len(self.d)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        xs = self.d[index].strip().split(' ')
        xs = [int(i.split(':')[0]) for i in xs[1:]]
        indices = np.array(xs)
        values = np.ones(len(xs))
        return indices,values
    def __get_handle_spm__(self):
        return self.d


class SimpleDataset(data.Dataset):
    def __init__(self, X_file):
        super(SimpleDataset, self).__init__()
        self.d = skds.load_svmlight_file(X_file)[0]
        #self.d = self.d[:,0:1000]
        self.length = self.d.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.d.getrow(index).tocoo()
        indices = np.array(x.col)
        #indices = indices + 1 # IT IS SHIFTED we want 1 based. 0 is used for invalid index . Will have to make changes for when 0 is one of the values
        values = np.array(x.data)
        return indices,values
    def __get_handle_spm__(self):
        return self.d



def get_dataset(tfile):
    return SimpleDataset(tfile)

def get_dna_dataset(tfile):
    return SimpleDNADataset(tfile)


if __name__ == '__main__':
    dataset = SimpleDNADataset("/home/apd10/experiments/projects/CompressCovariance/webspam/train.txt")
    dataset[5]
    pdb.set_trace()
