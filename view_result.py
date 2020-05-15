import pandas as pd  
import numpy as np   
import sys
import matplotlib.pyplot as plt
import argparse
import pdb

args = argparse.ArgumentParser()
args.add_argument("--files", type=str, action="store", dest="files", required=True)
args.add_argument("--savefig", type=str, action="store", dest="savefig", default=None)

results = args.parse_args()
FILES = results.files
SAVEFIG = results.savefig

def getlabel(f):
    xs =  f.split('/')[-1].split('_')
    return xs[-1] + '_' + xs[-2] + '_'  + xs[-3]


f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]})
once = False
for FILE in FILES.split(','):
        d = pd.read_csv(FILE, sep=",") 
        d['f1'] = 2*(d.precision * d.recall) / ((d.precision + d.recall))
        print("ActualIDSize CovTh F!Max")
        lens = []
        act_ths = []
        f1maxs = []
        d = d.dropna()
        d.sort_values(['len_actual_ids'], inplace=True)

        lens = d.len_actual_ids.unique()[::-1]

        for l in lens:  
            act_cov = d[(d.len_actual_ids==l)]['act_th'].values[0] 
            f1max = np.max(d[d.len_actual_ids == l].f1.unique())
            print(l, act_cov, f1max) 
            act_ths.append(act_cov)
            f1maxs.append(f1max)


        if not once:
            a0.plot(lens, act_ths, label='ActCov')
            once = True
        a0.legend()
        a1.plot(lens, f1maxs, label='f1max'+getlabel(FILE))
        a1.legend()
        #ax.set_xlabel('TopK Sets of Actual Covariances')
if SAVEFIG is None:
  plt.show()
else:
  plt.savefig(SAVEFIG)
