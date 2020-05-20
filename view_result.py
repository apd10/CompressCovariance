import pandas as pd  
import numpy as np   
import sys
import matplotlib.pyplot as plt
import argparse
import pdb

args = argparse.ArgumentParser()
args.add_argument("--files", type=str, action="store", dest="files", required=True)
args.add_argument("--savefig", type=str, action="store", dest="savefig", default=None)
args.add_argument("--min_sig", type=float, action="store", dest="min_sig", default=0)
args.add_argument("--max_sig", type=float, action="store", dest="max_sig", default=1.1)
args.add_argument("--label", type=str, action="store", dest="label", default="9,10,11,12,13,-1")


results = args.parse_args()
FILES = results.files
SAVEFIG = results.savefig
MIN_SIG = results.min_sig
MAX_SIG = results.max_sig

LABELS = [int(i) for i in results.label.split(',')]

def getlabel(f):
    xs =  f.split('/')[-1].split('_')
    #return xs[5] + '_' + xs[-1] + '_' + xs[-2] + '_'  + xs[-3]
    lbl = xs[LABELS[0]]
    for l in LABELS[1:]:
      if l < len(xs):
         lbl = lbl + '_' + xs[l]
    return lbl.replace("EXP0:0", "BASE")

f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]})
once = False
for FILE in FILES.split(','):
        print(FILE)
        d = pd.read_csv(FILE, sep=",") 
        d['f1'] = 2*(d.precision * d.recall) / ((d.precision + d.recall))
        #print("ActualIDSize CovTh F!Max")
        lens = []
        act_ths = []
        f1maxs = []
        d = d.dropna()
        d.sort_values(['len_actual_ids'], inplace=True)

        lens = d.len_actual_ids.unique()[::-1]

        for l in lens:  
            act_cov = d[(d.len_actual_ids==l)]['act_th'].values[0] 
            f1max = np.max(d[d.len_actual_ids == l].f1.unique())
            #print(l, act_cov, f1max) 
            act_ths.append(act_cov)
            f1maxs.append(f1max)
        f1maxs = np.array(f1maxs)
        act_ths = np.array(act_ths)
        mask = np.multiply(act_ths >= MIN_SIG, act_ths <= MAX_SIG)
        if not once:
            a0.plot(lens[mask], act_ths[mask], label='Actual Signal Values')
            #ticks  = (act_ths * 1000).astype(int) / 10.0
            #plt.xticks(lens[mask], ticks[mask])
            once = True
        a0.legend()
        a1.plot(lens[mask], f1maxs[mask], label=getlabel(FILE))
        a1.legend()
a1.set_xlabel('Top number of actual signals')
a0.set_ylabel('Actual Signal')
a1.set_ylabel('Max F1 score obtained')
a0.grid()
a1.grid()
if SAVEFIG is None:
  plt.show()
else:
  plt.savefig(SAVEFIG)
