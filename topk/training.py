import torch
torch.manual_seed(0)
from torch import nn
import torch.multiprocessing as mp
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from os import path
import os

from dataset import get_dataset
import pdb
import argparse
import time
from os.path import dirname, abspath, join
import glob
from Sketch import *
from tqdm import tqdm



parser = argparse.ArgumentParser()
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
parser.add_argument('--threshold.infer2.sig_pct', action="store" , dest="threshold_infer2_sig_pct", required=False, type=float, default=None,
                  help="constant threshold specified from outside")
parser.add_argument('--threshold.infer2.init_pct', action="store" , dest="threshold_infer2_init_pct", required=False, type=float, default=None,
                  help="constant threshold specified from outside")
parser.add_argument('--threshold.infer2.inexp_frac', action="store" , dest="threshold_infer2_inexp_frac", required=False, type=float, default=None,
                  help="constant threshold specified from outside")
parser.add_argument('--device_id', action="store", dest="device_id", default=0, type=int,
                    help="device gpu id")
results = parser.parse_args()
DATASET = results.dataset
INSERT = results.insert # default correlation
CS_REP = results.cs_rep
CS_RANGE = results.cs_range
MU_SAMPLES = results.use_samples_mu
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
THRESHOLD_INFER2_SIG_PCT = results.threshold_infer2_sig_pct
THRESHOLD_INFER2_INIT_PCT = results.threshold_infer2_init_pct
THRESHOLD_INFER2_INEXP_FRAC = results.threshold_infer2_inexp_frac
BATCH_SIZE = results.batch
device_id = results.device_id

assert(THRESHOLD_METHOD == "constant")
BATCH=1

def my_collate(batch):
    l = -1
    for d in batch:
      l = max(l, len(d[0]))
    indices = []
    values = []
    for d in batch:
      indices.append(np.append(d[0], np.zeros(l - len(d[0]))))
      values.append(np.append(d[1], np.zeros(l - len(d[1]))))
    indices_tensor = torch.LongTensor(np.array(indices))
    values_tensor = torch.FloatTensor(np.array(values))
    if device_id != -1:
      indices_tensor = indices_tensor.cuda(device_id)
      values_tensor = values_tensor.cuda(device_id)
    return [indices_tensor, values_tensor]


def train(data_file, countsketch, max_d):
    global THRESHOLD_CONST_EXP
    # ===========================================================
    # Prepare train dataset & test dataset
    # ===========================================================
    print("***** prepare data ******")

    data_set = get_dataset(data_file)
    total_samples = data_set.__len__() - MU_SAMPLES
    train_dataloader = torch.utils.data.DataLoader(dataset=data_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
    running_mu_sum = torch.zeros(max_d)
    running_mu = torch.zeros(max_d)
    running_mu2_sum = torch.zeros(max_d)
    running_mu2 = torch.zeros(max_d)
    running_std = torch.zeros(max_d)
    if device_id != -1:
        running_mu_sum,running_mu,running_mu2_sum,running_mu2,running_std = running_mu_sum.cuda(device_id),running_mu.cuda(device_id),running_mu2_sum.cuda(device_id),running_mu2.cuda(device_id),running_std.cuda(device_id)
    num_samples = 0

    if THRESHOLD_METHOD == "constant":
      if THRESHOLD_CONST_EXP_FRAC is not None:
        THRESHOLD_CONST_EXP = int(THRESHOLD_CONST_EXP_FRAC * total_samples)
        print("EXPLORATION PERIOD", THRESHOLD_CONST_EXP,"/",total_samples)
      exploration_samples = THRESHOLD_CONST_EXP
      init_threshold = THRESHOLD_CONST_THOLD
      theta = THRESHOLD_CONST_THETA

    ignored = 0
    for iteration, (indices, values) in tqdm(enumerate(train_dataloader), total=1000):
        #indices batch x features
        #values batch x features
        print(iteration, indices.shape)
        if indices.shape[1] > 9000:
          ignored = ignored + 1
          print("IGNORED", ignored)
          total_samples = total_samples -1
          continue

        # update the mu and std
        num_samples =  (iteration + 1) * BATCH_SIZE
        flat_index = indices.reshape(1,-1).squeeze()
        flat_values = values.reshape(1,-1).squeeze()
        running_mu_sum.scatter_add_(0, flat_index, flat_values)
        running_mu2_sum.scatter_add_(0, flat_index, flat_values**2)
        running_mu = running_mu_sum / num_samples
        running_mu2 = running_mu2_sum / num_samples
        running_std = torch.sqrt(running_mu2 - running_mu ** 2)  + 1e-6
        
        if INSERT == "correlation":
            # NOTE we are computing E(XY/(std(X) std(Y))) if we were to do -mu,  the data is not sparse
            values = values  / running_std[indices]
        values = values / total_samples

        if iteration * BATCH_SIZE < MU_SAMPLES:
            # phase 1
            continue
        elif iteration * BATCH_SIZE < MU_SAMPLES + exploration_samples:
            # phase 2
            # we only add to count sketch without any sampling
            countsketch.insert(indices, values, None)
            continue
        else:
            # insert with thold
            thold = init_threshold + (num_samples - exploration_samples - MU_SAMPLES) / total_samples * theta
            countsketch.insert(indices, values, thold)
    print("IGNORED", ignored)

def dump_topk(countsketch):
    pdb.set_trace()

if __name__ == '__main__':
    datafile = '/home/apd10/experiments/projects/CompressCovariance/' + DATASET + '/train.txt'
    max_d = 17_000_000
    topK = 1000
    countsketch = CountSketch(CS_REP, CS_RANGE, max_d, topK, device_id)
    train(datafile, countsketch, max_d)
    dump_topk(countsketch)