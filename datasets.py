import scipy
import sklearn.datasets as skds
from os.path import dirname, abspath, join
import pdb
cur_dir = dirname(abspath(__file__))
import numpy as np

def get_dataset(dataset):
  path = join(cur_dir, dataset, "train.txt")
  data = skds.load_svmlight_file(path)
  return data

  
