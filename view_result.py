import pandas as pd  
import numpy as np   
import sys
import matplotlib.pyplot as plt
import argparse

args = argparse.ArgumentParser()
args.add_argument("--files", type=str, action="store", dest="files", required=True)
args.add_argument("--savefig", type=str, action="store", dest="savefig", default=None)

results = args.parse_args()
FILES = results.files
SAVEFIG = results.savefig

def getlabel(f):
    return f.split('/')[-1].split('_')[-1]

for FILE in FILES.split(','):
        d = pd.read_csv(FILE, sep=",") 
        d['f1'] = 2*(d.precision * d.recall) / ((d.precision + d.recall))
        print("ActualIDSize CovTh F!Max")
        lens = []
        act_ths = []
        f1maxs = []

        lens = d.len_actual_ids.unique()[::-1]

        for l in lens:  
            act_cov = d[(d.len_actual_ids==l)]['act_th'].values[0] 
            f1max = np.max(d[d.len_actual_ids == l].f1.unique())
            print(l, act_cov, f1max) 
            act_ths.append(act_cov)
            f1maxs.append(f1max)


        plt.subplot(211)
        plt.plot(lens, act_ths, label='ActCov'+getlabel(FILE))
        plt.legend()
        ax = plt.subplot(212)
        plt.plot(lens, f1maxs, label='f1max'+getlabel(FILE))
        plt.legend()
        ax.set_xlabel('TopK Sets of Actual Covariances')
if SAVEFIG is None:
  plt.show()
else:
  plt.savefig(SAVEFIG)
