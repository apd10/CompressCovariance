import numpy as np
import pdb
import argparse
from os.path import dirname, abspath, join
from tqdm import tqdm
import pickle
import sklearn.datasets as skds


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action="store", dest="dataset", type=str, required=True,
                    help="Dataset")
parser.add_argument('--pickle', action="store", dest="pickle_file", required=True, type=str,
                    help="pickle file to evaluate")

results = parser.parse_args()
DATASET = results.dataset
PICKLE_FILE = results.pickle_file

max_d = 17_000_000


def get_ij(x):
    return x//max_d, x%max_d


def eval(dic, X):
    s = 0
    lines = []
    ip = 0
    for key in dic.keys():
        i,j = get_ij(key)
        a = np.array(X[:,i].todense()).reshape(1,X.shape[0]) 
        b = np.array(X[:,j].todense()).reshape(1,X.shape[0])
        if np.std(a) > 0 and np.std(b) > 0:
            cov = np.corrcoef(a,b)[0,1]
            cov = np.abs(cov)
            s = s + cov
            ip = ip + 1
            print(key, i, j, cov)
            if ip % 500 == 0:
                lines.append('{} - {} {} {} cs: {} act: {} avg: {}\n'.format(ip, key, i, j, dic[key], cov, s/ip))
    
    print(lines)



if __name__ == '__main__':
    datafile = '/home/apd10/experiments/projects/CompressCovariance/' + DATASET + '/train.txt'

    with open(PICKLE_FILE, "rb") as f:
        dic = pickle.load(f)
    values =  [ i for i in dic.values()]
    print(np.min(values), np.max(values), np.mean(values))
    
    print("Loading")
    X = skds.load_svmlight_file("../url/train.txt")[0]
    print("Loaded")
    X = X.tocoo().tocsc()
    print("converted")
    eval(dic, X)

