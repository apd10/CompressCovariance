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
import concurrent
from design_thresholds import find_best_exploration_period, find_best_theta, find_best_exploration_period_m2


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action="store", dest="dataset", type=str, required=True,
                    help="Dataset")
parser.add_argument('--insert', action="store", default="correlation",
                    help="insert all pairs explicitly")
parser.add_argument('--countsketch.repetitions', action="store", dest="cs_rep", required=True, type=int,
                   help="number of repititions")
parser.add_argument('--countsketch.range', action="store", dest="cs_range", required=True, type=int,
                   help="range of each array")
parser.add_argument('--insert.samples_for_mu', action="store", dest="use_samples_mu", required=True, type=int,
                    help="use first few samples mentioned here for mu computation . then start inserting")
parser.add_argument('--use_first_K_features', action="store" , dest="num_features_to_use", required=False, type=int, default=1000,
                  help="We will use first K features of the dataset for complete evaluation")
parser.add_argument('--batch', action="store" , dest="batch", required=False, type=int, default=None,
                  help="Batch")
parser.add_argument('--alpha', action="store" , dest="alpha", required=False, type=float, default=0.001,
                  help="expected alpha. fraction of signals for plotting")
parser.add_argument('--signal', action="store" , dest="signal", required=False, type=float, default=0.25,
                  help="expected alpha. fraction of signals for plotting")
parser.add_argument('--threshold_method', action='store', dest='threshold_method', required=False, type=str, default=None,
                  help="\n constant: specify a constant threshold in threshold.const\
                        \n infer_a1: specify values of alpha, sigma in threshold.alpha and threshold.sigma")
parser.add_argument('--threshold.const.thold', action="store" , dest="threshold_const_thold", required=False, type=float, default=None,
                  help="constant threshold specified from outside")
parser.add_argument('--threshold.const.theta', action="store" , dest="threshold_const_theta", required=False, type=float, default=None,
                  help="constant threshold specified from outside")
parser.add_argument('--threshold.const.exp', action="store" , dest="threshold_const_exp", required=False, type=int, default=None,
                  help="constant threshold specified from outside")
parser.add_argument('--threshold.const.exp_frac', action="store" , dest="threshold_const_exp_frac", required=False, type=float, default=None,
                  help="constant threshold specified from outside")
parser.add_argument('--threshold.infer.thold', action="store" , dest="threshold_infer_thold", required=False, type=float, default=None,
                  help="constant threshold specified from outside")
parser.add_argument('--use_num_samples', action="store", dest="use_num_samples", required=False, type=int, default=None, 
                  help="total samples to use for estimation from countsketch ")
parser.add_argument('--target_prob1', action="store" , dest="target_prob1", required=False, type=float, default=None,
                  help="miss at exploration ")
parser.add_argument('--target_prob2', action="store" , dest="target_prob2", required=False, type=float, default=None,
                  help="miss at total_samples ")
parser.add_argument('--run_base', action="store_true" , required=False, default=False,
                  help="run base cs ")
parser.add_argument('--filter', action="store_true" , required=False, default=False,
                  help="Do a filtered evaluation for values which have better confidence intervals")
results = parser.parse_args()
DATASET = results.dataset
INSERT = results.insert # default correlation
CS_REP = results.cs_rep
CS_RANGE = results.cs_range
MU_SAMPLES = results.use_samples_mu
NUM_FEATURES = results.num_features_to_use
BATCH = results.batch
THRESHOLD_METHOD = results.threshold_method
THRESHOLD_CONST_THOLD = results.threshold_const_thold
THRESHOLD_INFER_THOLD = results.threshold_infer_thold
THRESHOLD_CONST_THETA = results.threshold_const_theta
THRESHOLD_CONST_EXP = results.threshold_const_exp
THRESHOLD_CONST_EXP_FRAC = results.threshold_const_exp_frac
ALPHA = results.alpha
SIGNAL = results.signal
USE_NUM_SAMPLES = results.use_num_samples
TARGET_PROB1 = results.target_prob1
TARGET_PROB2 = results.target_prob2
RUN_BASE = results.run_base
FILTER = results.filter

if BATCH is None:
  if DATASET == "gisette":
    BATCH = 10
  if DATASET == "rcv1":
    BATCH = 100
  if DATASET == "news20":
    BATCH = 100
  if DATASET == "sector":
    BATCH = 100

if RUN_BASE:
  print("Running BASE")
  THRESHOLD_METHOD = "constant"
  THRESHOLD_CONST_THOLD = 0
  THRESHOLD_CONST_THETA = 0
  THRESHOLD_CONST_EXP = 0
assert(THRESHOLD_METHOD is not None)
print("Threshold", THRESHOLD_METHOD)
if THRESHOLD_METHOD == "constant":
  assert(THRESHOLD_CONST_THOLD is not None)
  assert(THRESHOLD_CONST_EXP is not None or THRESHOLD_CONST_EXP_FRAC is not None)
elif THRESHOLD_METHOD == "infer":
  assert(THRESHOLD_INFER_THOLD is not None)
  assert(SIGNAL is not None)
  
    
filekey = 'INS{}_CRG{}_CRP{}_MUSMP{}_TS{}_NUMFEAT{}_BT{}_ALPHA{}_METHOD{}_TH{}_THETA{}_EXP{}:{}_FILT{}'.format(INSERT,CS_RANGE,CS_REP,MU_SAMPLES, USE_NUM_SAMPLES, NUM_FEATURES, BATCH, ALPHA, THRESHOLD_METHOD, THRESHOLD_CONST_THOLD, THRESHOLD_CONST_THETA, THRESHOLD_CONST_EXP ,THRESHOLD_CONST_EXP_FRAC, FILTER)

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
  print("Suggested Signal: 99.5%-ile ", np.percentile(series, 0.995*100))
  print("Suggested init_thold: 50%-ile ", np.percentile(series, 0.5*100))
      

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


def sketch_data(data, countsketch, batch):
  global filekey
  features = data.shape[1]
  _indices = np.triu_indices(features, k=1)
  num_entries_in_cs = len(_indices[0])
  num_data = data.shape[0]

  total_samples = num_data - MU_SAMPLES
  running_mu_sum = np.zeros(features)
  running_mu = np.zeros(features)
  running_mu2 = np.zeros(features)
  running_std = np.zeros(features)

  # [0:MU_SAMPLES]
  BASE = 0
  running_mu_sum = np.sum(data[0:MU_SAMPLES,:].todense(), axis=0) # 1,features E(x)
  running_mu2_sum = np.sum(np.square(data[0:MU_SAMPLES,:].todense()), axis=0) #E(x2)
  running_mu = running_mu_sum / MU_SAMPLES
  running_mu2 = running_mu2_sum / MU_SAMPLES
  running_std = np.sqrt(running_mu2 - np.square(running_mu))  + 1e-6
  #pdb.set_trace()

  if THRESHOLD_METHOD == "constant":
    if THRESHOLD_CONST_EXP_FRAC is not None:
      THRESHOLD_CONST_EXP = int(THRESHOLD_CONST_EXP_FRAC * total_samples)
      print("EXPLORATION PERIOD", THRESHOLD_CONST_EXP,"/",total_samples)
    exploration_samples = THRESHOLD_CONST_EXP
    init_threshold = THRESHOLD_CONST_THOLD
    theta = THRESHOLD_CONST_THETA

  elif THRESHOLD_METHOD == "infer":
    init_threshold = THRESHOLD_INFER_THOLD
    exploration_samples = find_best_exploration_period_m2(SIGNAL, ALPHA, init_threshold, num_entries_in_cs, {"K": CS_REP , "R": CS_RANGE}, total_samples, TARGET_PROB1) # cannot get below 0.07 it seems
    print("exploration_samples", exploration_samples, "/", total_samples)
    exploration_samples = min(exploration_samples, int(0.95*total_samples))
    print("exploration_samples", exploration_samples, "/", total_samples)
    theta = find_best_theta(SIGNAL, ALPHA, init_threshold, num_entries_in_cs, {"K": CS_REP , "R": CS_RANGE}, total_samples, exploration_samples, TARGET_PROB2) # 
    print("theta", theta)
    filekey = 'INS{}_CRG{}_CRP{}_MUSMP{}_TS{}_NUMFEAT{}_BT{}_ALPHA{}_METHOD{}_TH{}_THETA{}_EXP{}_TPROB1{}_TPROB2{}_SIGNAL{}_FILTER{}'.format(INSERT,CS_RANGE,CS_REP,MU_SAMPLES, USE_NUM_SAMPLES, NUM_FEATURES, BATCH, ALPHA, THRESHOLD_METHOD, THRESHOLD_INFER_THOLD, int(theta*1000)/1000.0, exploration_samples, TARGET_PROB1, TARGET_PROB2, SIGNAL, FILTER)
    print("Updated filekey")
    print(filekey)

    
  
  # [ MU_SAMPLES : MU_SAMPLES + EXPLORATION ]

  #std = np.std(data[MU_SAMPLES:,:].todense(), axis=0)
  #mu = np.mean(data[MU_SAMPLES:,:].todense(), axis=0)
  if exploration_samples > 0:
    BASE = MU_SAMPLES
    low = BASE
    high = BASE + exploration_samples
    dense = data[low:high, :].todense()
    running_mu_sum = running_mu_sum + np.sum(dense, axis=0)
    running_mu2_sum = running_mu2_sum + np.sum(np.square(dense), axis=0)
    running_mu = running_mu_sum / high
    running_mu2 = running_mu2_sum / high
    running_std = np.sqrt(running_mu2 - np.square(running_mu)) + 1e-6
  
    if INSERT == "correlation":
        print("Correlation")
        #shifted = (dense - mu) / std
        shifted = (dense - running_mu) / running_std
    else:
        shifted = (dense - running_mu)
  
    crentries = np.matmul(shifted.transpose(), shifted) / total_samples
    crentries = crentries[_indices]
    crentries = np.array(crentries).reshape(crentries.size,)
  
    # get triu entries
    countsketch.insert_all(crentries)
  
  # [MU_SAMPLES + EXPLORATION:]
  BASE = MU_SAMPLES + exploration_samples
  num_batches = int(total_samples / batch) + 1
  for i in tqdm(range(0, num_batches)):
    low = BASE + i * batch
    high = min(low + batch, num_data)
    dense = data[low:high, :].todense()
    running_mu_sum = running_mu_sum + np.sum(dense, axis=0)
    running_mu2_sum = running_mu2_sum + np.sum(np.square(dense), axis=0)
    running_mu = running_mu_sum / high
    running_mu2 = running_mu2_sum / high
    running_std = np.sqrt(running_mu2 - np.square(running_mu)) + 1e-6
    if INSERT == "correlation":
        shifted = (dense - running_mu) / running_std
        #shifted = (dense - mu) / std
    else:
        shifted = (dense - running_mu)

    crentries = np.matmul(shifted.transpose(), shifted) / total_samples # (N,N)
    crentries = crentries[_indices]
    crentries = np.array(crentries).reshape(crentries.size,)
    # get triu
    crestimates = countsketch.query_all()
    thold = init_threshold + theta * (low - BASE)/total_samples  #Tau + theta*(t_0-t)/T # can be high / low . keeping low to match zhenweis code
    mask = np.abs(crestimates) >= thold
    #print(i, thold, np.sum(mask), "/", len(mask))
    crentries = np.multiply(crentries, mask) # here we select the potential signals
    countsketch.insert_all(crentries)



covariances = None
cs_covariances = None

def eval_ind(qt, pt):
  act_th = np.quantile(covariances, qt)
  actual_idxs = np.argwhere(covariances >= act_th)
  actual_idxs = actual_idxs.reshape(actual_idxs.size,)
  pred_th = np.quantile(cs_covariances, pt)
  pred_idxs = np.argwhere(cs_covariances >= pred_th)
  pred_idxs = pred_idxs.reshape(pred_idxs.size,)
  if len(pred_idxs) > 0 and len(actual_idxs) > 0:
    recall = get_recall(actual_idxs, pred_idxs)
    precision = get_precision(actual_idxs, pred_idxs)
  else:
    recall = 0
    precision = 0
  return qt,act_th,pt,pred_th,recall,precision,len(actual_idxs),len(pred_idxs)


def evaluate(data, countsketch, record_dir, filekey):
  global covariances
  global cs_covariances

  # compute all the covariances from all the data
  features = data.shape[1]
  _indices = np.triu_indices(features, k=1)
  num_data = data.shape[0]
  total_samples = num_data - MU_SAMPLES

  # computing actual covariances
  densedata = data[MU_SAMPLES:, :].todense()
  mu = np.mean(densedata, axis=0) #(1,NUM_FEATURES)
  std = np.std(densedata, axis=0) + 1e-6

  if INSERT == "correlation":
    shifted = (densedata - mu) / std
  else:
    shifted = densedata - mu

  covariances = np.matmul(shifted.transpose(), shifted) / total_samples
  covariances = covariances[_indices]
  covariances = np.array(covariances).reshape(covariances.size,)

  cs_covariances = countsketch.query_all()


  covariances = np.abs(covariances)
  cs_covariances = np.abs(cs_covariances)

  print(np.corrcoef(covariances, cs_covariances))
  print("Stats Actual")
  printStats(covariances)
  print("Stats CS")
  printStats(cs_covariances)
  

  sparsity=2*ALPHA
  qtile = np.arange(1-sparsity, 1, sparsity/40)
  ptile = np.arange(1-2*sparsity, 1, sparsity/40)
  qptiles = np.transpose([np.tile(qtile, len(ptile)), np.repeat(ptile, len(qtile))])
  dump_values = []
  dump_values.append("{},{},{},{},{},{},{},{}\n".format("qt","act_th","pt","pred_th","recall","precision", "len_actual_ids", "len_pred_ids"))
  with concurrent.futures.ProcessPoolExecutor(20) as executor:
      futures = []
      print("submitting jobs")
      for i in tqdm(range(0, qptiles.shape[0])):
        qtile = qptiles[i][0]
        ptile = qptiles[i][1]
        futures.append(executor.submit(eval_ind, qtile, ptile))
      ip = 0
      print("waiting for executions")
      for res in tqdm(concurrent.futures.as_completed(futures), total=qptiles.shape[0]):
        ip = ip + 1
        qt,act_th,pt,pred_th,recall,precision,lenA,lenP = res.result()
        dump_values.append("{},{},{},{},{},{},{},{}\n".format(qt,act_th,pt,pred_th,recall,precision,lenA,lenP))

  fname = join(record_dir, "data_"+filekey)
  with open(fname, "w") as f:
    f.writelines(dump_values)
  
if __name__ == '__main__':
  # return sparse dataset in coo format
  record_dir = join(cur_dir, DATASET, "record")
  print(filekey)

  data = get_dataset(DATASET)[0]
  if DATASET == "gissette":
    data.data = data.data + 1
  if FILTER:
    print("FILTER IS SET!")
    print(" Total number of features before filtering", data.shape[1])
    x = data.tocoo().col
    x = np.sort(x)
    chosen = np.zeros(data.shape[1]) 
    idx = 0 
    for i in range(0, len(chosen)): 
        ct = 0 
        while(idx < len(x) and x[idx] == i): 
            ct = ct + 1 
            idx = idx + 1 
        chosen[i] = ct 
        if idx >= len(x): 
            break 
    mask = chosen > 0.01 * data.shape[0]
    print("After filtering", np.sum(mask))
    a = np.argwhere(mask) 
    a = a.reshape(a.size,)
    data = data[:, a]
    print(" Total number of features after filtering", data.shape[1])

  np.random.seed(101)
  sff = np.arange(0, data.shape[0])
  np.random.shuffle(sff)
  data = data[sff]
  if USE_NUM_SAMPLES is not None:
    data = data[0:USE_NUM_SAMPLES,:]
  original_features = data.shape[1]
  features = np.random.randint(0, original_features, NUM_FEATURES)
  data = data[:,features] # keep only the first few features
  _indices = np.triu_indices(NUM_FEATURES, k=1)
  countsketch = CountSketch(CS_REP, CS_RANGE, len(_indices[0]))
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
