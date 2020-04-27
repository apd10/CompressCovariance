import numpy as np
from os import path
import os
from os.path import dirname, abspath, join
import glob
cur_dir = dirname(abspath(__file__))
from Sketch import CountSketch
from datasets import get_dataset
import pdb
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action="store", dest="dataset", type=str, required=True,
                    help="Dataset")
parser.add_argument('--insertall', action="store_true", default=True,
                    help="insert all pairs explicitly")
parser.add_argument('--verifyall', action="store_true", default=True,
                    help="verifyall: do entire precision recall analysis")
parser.add_argument('--countsketch.repetitions', action="store", dest="cs_rep", required=True, type=int,
                   help="number of repititions")
parser.add_argument('--countsketch.range', action="store", dest="cs_range", required=True, type=int,
                   help="range of each array")
parser.add_argument('--insertall.skip_samples_for_mu', action="store", dest="use_samples_mu", required=True, type=int,
                    help="use first few samples mentioned here for mu computation . then start inserting")

results = parser.parse_args()
DATASET = results.dataset
INSERTALL = results.insertall
VERIFYALL = results.verifyall
CS_REP = results.cs_rep
CS_RANGE = results.cs_range
MU_SAMPLES = results.use_samples_mu


def get_id(i, j, num_features):
  return i * num_features + j

def get_recall(true_set, pred_set):
  ts = set(true_set)
  ps = set(pred_set)
  ints = ts & ps
  return len(ints) / len(ts)

def get_precision(true_set, pred_set):
  ts = set(true_set)
  ps = set(pred_set)
  ints = ts & ps
  return len(ints) / len(ps)

def sketch_data_insertall(data, countsketch):
  features = data.shape[1]
  num_data = data.shape[0]
  mu_sum = np.zeros(features) # while using divide by num_samples processed
  normalizer = num_data - MU_SAMPLES

  # bootstrapping MU_SAMPLES
  for i in range(0, MU_SAMPLES):
    row = data.getrow(i)
    _,c = row.nonzero()
    for cx in c:
      mu_sum[cx] = mu_sum[cx] + row[0,cx]
  data_to_insert = np.zeros(features*features)

  for i in range(MU_SAMPLES, num_data):
    row = data.getrow(i)
    _,c = row.nonzero()
    for cx in c:
      mu_sum[cx] = mu_sum[cx] + row[0,cx]
    mu = mu_sum / (i+1)
    for x in range(0, len(c)):
      cx = c[x]
      vx = row[0, cx] - mu[cx]
      for y in range(0, x):
        #print(x,y)
        cy = c[y]
        vy  = row[0, cy] - mu[cy]
        idx = get_id(cx, cy, features)
        value = vx*vy / normalizer
        data_to_insert[idx] = data_to_insert[idx] + value
  for i in range(0, len(data_to_insert)):
    countsketch.insert(i, value)
  return mu_sum / num_data

def sketch_data(data, countsketch):
  # insert all data in the sketch
  if INSERTALL:
    mu = sketch_data_insertall(data, countsketch)
  return mu


def evaluate_verifyall(data, countsketch, mu, record_dir, filekey):
  # compute all the covariances from all the data
  features = data.shape[1]
  num_data = data.shape[0]
  normalizer = num_data
  covariances = np.zeros(features * features)

  for i in range(0, num_data):
    row = data.get_row(i)
    _,c = row.nonzero()
    for cx in c:
      vx = row[0, cx] - mu[cx]
      for cy in c:
        vy  = row[0, cy] - mu[cy]
        idx = get_id(cx, cy, features)
        value = vx*vy / normalizer
        covariances[idx] = covariances[idx]
  # now we have all the variances and covariances
  # we will evaluate this at different quantiles of the actual covariances
  cs_covariances = np.zeros(features*features)
  for idx in range(0, len(covariances)):
    cs_covariances[idx] = countsketch.query(idx)
  

  qtile = np.arange(0.5, 1, 0.1)
  dump_values = []
  dump_values.append("{},{},{},{},{},{}".format("qt","act_th","pt","pred_th","recall","precision"))
  for qt in qtile:
    act_th = np.quantile(covariances, qt)
    actual_idxs = np.argwhere(covariances > act_th)
    actual_idxs = actual_idxs.reshape(actual_idxs.size,)
    for pt in np.arange(0, 1, 0.05):
      pred_th = np.quantile(cs_covariances, pt)
      pred_idxs = np.argwhere(cs_covariances > pred_th)
      pred_idxs = pred_idxs.reshape(pred_idxs.size,)
      recall = get_recall(actual_idxs, pred_idxs)
      precision = get_precision(actual_idxs, pred_idxs)
      dump_values.append("{},{},{},{},{},{}".format(qt,act_th,pt,pred_th,recall,precision))
  fname = join(record_dir, "data_"+filekey)
  with open(fname, "w") as f:
    f.writelines(dump_values)
  
      
def evaluate(data, countsketch, mu, record_dir, filekey):
  if VERIFYALL:
    evaluate_verifyall(data, countsketch, mu, record_dir, filekey)

if __name__ == '__main__':
  # return sparse dataset in coo format
  record_dir = join(cur_dir, DATASET, "record")
  filekey = 'INS{}_VER{}_CRG{}_CRP{}_MUSMP{}'.format(INSERTALL,VERIFYALL,CS_RANGE,CS_REP,MU_SAMPLES)

  data = get_dataset(DATASET)[0]
  countsketch = CountSketch(CS_REP, CS_RANGE)
  mu = sketch_data(data, countsketch)
  evaluate(data, countsketch, mu, record_dir, filekey)
