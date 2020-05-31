import torch
from torch.utils import data
import numpy as np
from sklearn.utils import murmurhash3_32
import pdb
import pandas as pd
import sklearn.datasets as skds


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
        values = np.array(x.data)
        return indices,values
    def __get_handle_spm__(self):
        return self.d



def get_dataset(tfile):
    return SimpleDataset(tfile)


if __name__ == '__main__':
    dataset = SimpleDataset("/home/apd10/experiments/projects/CompressCovariance/webspam/train.txt")
    dataset[5]
    pdb.set_trace()
