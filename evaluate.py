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
parser.add_argument('--use_first_K_features', action="store" , dest="num_features_to_use", required=False, type=int, default=1000,
                  help="We will use first K features of the dataset for complete evaluation")
parser.add_argument('--batch', action="store" , dest="batch", required=False, type=int, default=1000,
                  help="Batch")
parser.add_argument('--threshold', action="store" , dest="threshold", required=False, type=float, default=None,
                  help="threshold")
parser.add_argument('--sparsity', action="store" , dest="sparsity", required=False, type=float, default=0.001,
                  help="expected sparsity. fraction of signals for plotting")

results = parser.parse_args()
DATASET = results.dataset
INSERTALL = results.insertall
VERIFYALL = results.verifyall
CS_REP = results.cs_rep
CS_RANGE = results.cs_range
MU_SAMPLES = results.use_samples_mu
NUM_FEATURES = results.num_features_to_use
BATCH = results.batch
THRESHOLD = results.threshold
SPARSITY = results.sparsity
print("Value of Threshold", THRESHOLD)

def printStats(series):
  print('Mean:  Min  Max  10%ile  20%ile  30%iile  40%ile  50%ile  60ile  70ile  80ile  90ile  95ile  99ile  99.99  99.999')
  values = []
  for a in np.arange(0, 1, 0.1):
    values.append(np.percentile(series, a*100))
  values.append(np.percentile(series, 0.95*100))
  values.append(np.percentile(series, 0.99*100))
  values.append(np.percentile(series, 0.9999*100))
  values.append(np.percentile(series, 0.99999*100))
  print(np.mean(series), np.min(series), np.max(series), values)
      

def get_id(i, j):
  return i * NUM_FEATURES + j

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


def sketch_data_insertall(data, countsketch, batch):
  features = data.shape[1]
  num_data = data.shape[0]
  normalizer = num_data - MU_SAMPLES
  running_mu_sum = np.zeros(features)
  running_mu = np.zeros(features)
  running_mu_sum = np.sum(data[0:MU_SAMPLES,:].todense(), axis=0) # 1,features
  
  num_batches = int(normalizer / batch) + 1
  for i in tqdm(range(0, num_batches)):
    low = MU_SAMPLES + i * batch
    high = min(low + batch, num_data)
    dense = data[low:high, :].todense()

    running_mu_sum = running_mu_sum + np.sum(dense, axis=0)
    running_mu = running_mu_sum / high
    shifted = dense - running_mu 
    cventries = np.matmul(shifted.transpose(), shifted) 
    if THRESHOLD is not None:
        print("non_zeros", np.sum(cventries != 0))
        cventries = np.multiply(cventries, (np.abs(cventries) > THRESHOLD * batch))
        print("post non_zeros", np.sum(cventries != 0))
    cventries = cventries / normalizer # features,features
    cventries = np.array(cventries).reshape(cventries.size)
    # need a parallel version for this
    #for idx in range(0, len(cventries)):
    #  if cventries[idx] != 0:
    #    countsketch.insert(idx, cventries[idx])
    countsketch.insert_all(cventries) # feat^2
 
  print("Run Mu",np.sum(running_mu))


def sketch_data(data, countsketch, batch):
  # insert all data in the sketch
  if INSERTALL:
    mu = sketch_data_insertall(data, countsketch, batch)


def evaluate_verifyall(data, countsketch, record_dir, filekey):
  # compute all the covariances from all the data
  features = data.shape[1]
  num_data = data.shape[0]
  normalizer = num_data
  # computing actual covariances
  densedata = data.todense()
  mu = np.mean(densedata, axis=0) #(1,NUM_FEATURES)
  print("Act Mu",np.sum(mu))
  shifted = densedata - mu
  covariances = np.matmul(shifted.transpose(), shifted) / (num_data - 1)
  covariances = np.array(covariances).reshape(covariances.size) # TODO check stacking

  cs_covariances = np.zeros(features*features)
  # we will need a parallel version for this
  for idx in tqdm(range(0, len(covariances))):
    cs_covariances[idx] = countsketch.query(idx)


  covariances = np.abs(covariances)
  cs_covariances = np.abs(cs_covariances)
  print(np.corrcoef(covariances, cs_covariances))
  print("Stats Actual")
  printStats(covariances)
  print("Stats CS")
  printStats(cs_covariances)

  sparsity=SPARSITY
  qtile = np.arange(1-sparsity, 1, sparsity/50)
  dump_values = []
  dump_values.append("{},{},{},{},{},{},{},{}\n".format("qt","act_th","pt","pred_th","recall","precision", "len_actual_ids", "len_pred_ids"))
  for qt in qtile:
    act_th = np.quantile(covariances, qt)
    actual_idxs = np.argwhere(covariances > act_th)
    actual_idxs = actual_idxs.reshape(actual_idxs.size,)
    if len(actual_idxs) == 0:
      continue
    for pt in np.arange(1-2*sparsity, 1, sparsity/50):
      pred_th = np.quantile(cs_covariances, pt)
      pred_idxs = np.argwhere(cs_covariances > pred_th)
      pred_idxs = pred_idxs.reshape(pred_idxs.size,)
      if len(pred_idxs) == 0:
        continue
      recall = get_recall(actual_idxs, pred_idxs)
      precision = get_precision(actual_idxs, pred_idxs)
      dump_values.append("{},{},{},{},{},{},{},{}\n".format(qt,act_th,pt,pred_th,recall,precision,len(actual_idxs),len(pred_idxs)))
  fname = join(record_dir, "data_"+filekey)
  with open(fname, "w") as f:
    f.writelines(dump_values)
  
      
def evaluate(data, countsketch, record_dir, filekey):
  if VERIFYALL:
    evaluate_verifyall(data, countsketch, record_dir, filekey)

if __name__ == '__main__':
  # return sparse dataset in coo format
  record_dir = join(cur_dir, DATASET, "record")
  filekey = 'INS{}_VER{}_CRG{}_CRP{}_MUSMP{}_NUMFEAT{}_BT{}_TH{}_SPSTY{}'.format(INSERTALL,VERIFYALL,CS_RANGE,CS_REP,MU_SAMPLES, NUM_FEATURES, BATCH, THRESHOLD, SPARSITY)
  print(filekey)

  data = get_dataset(DATASET)[0]
  np.random.seed(101)
  original_features = data.shape[1]
  features = np.random.randint(0, original_features, NUM_FEATURES)
  data = data[:,features] # keep only the first few features
  countsketch = CountSketch(CS_REP, CS_RANGE, NUM_FEATURES*NUM_FEATURES)
  sketch_data(data, countsketch, BATCH)
  evaluate(data, countsketch, record_dir, filekey)






'''
def sketch_data_insertall(data, countsketch):
  features = data.shape[1]
  num_data = data.shape[0]
  mu_sum = np.zeros(features) # while using divide by num_samples processed
  normalizer = num_data - MU_SAMPLES
  
  # bootstrapping MU_SAMPLES
  for i in range(0, MU_SAMPLES):
    row = data.getrow(i).tocoo()
    c = row.col
    d = row.data
    for x in range(0,len(c)):
      cx = c[x]
      mu_sum[cx] = mu_sum[cx] + d[x]
  data_to_insert = {}
  batch = 100
  for i in tqdm(range(MU_SAMPLES, num_data)):
    row = data.getrow(i).tocoo()
    c = row.col
    d = row.data
    for x in range(0,len(c)):
      cx = c[x]
      mu_sum[cx] = mu_sum[cx] + d[x]
    mu = mu_sum / (i+1)

    for x in range(0,len(c)):
      cx = c[x]
      vx = d[x] - mu[cx]
      for y in range(0,x):
        cy = c[y]
        vy  = d[y] - mu[cy]
        idx = get_id(cx, cy)
        value = vx*vy / normalizer
        if idx in data_to_insert.keys():
          data_to_insert[idx] = data_to_insert[idx] + value
        else:
          data_to_insert[idx] = value
    if i % batch == 0:
      for key in data_to_insert.keys():
        countsketch.insert(i, data_to_insert[key])
      data_to_insert = {}
  return mu_sum / num_data
'''
