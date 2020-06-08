import pandas as pd  
import numpy as np   
import sys
import matplotlib 
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
import matplotlib.pyplot as plt
import argparse
import pdb
from scipy.interpolate import interp1d
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


args = argparse.ArgumentParser()
args.add_argument("--files", type=str, action="store", dest="files", required=True)
args.add_argument("--savefig", type=str, action="store", dest="savefig", default=None)
args.add_argument("--min_sig", type=float, action="store", dest="min_sig", default=-1)
args.add_argument("--max_sig", type=float, action="store", dest="max_sig", default=10**6)
args.add_argument("--label", type=str, action="store", dest="label", default=None)
args.add_argument("--plot_ratio", action="store_true")
args.add_argument("--plot_all_signals", action="store_true")


results = args.parse_args()
FILES = results.files
SAVEFIG = results.savefig
MIN_SIG = results.min_sig
MAX_SIG = results.max_sig
PLOT_RATIO = results.plot_ratio
PLOT_ALL_SIGNALS = results.plot_all_signals
if results.label is not None:
    LABELS = [int(i) for i in results.label.split(',')]
else:
    LABELS = []
LENS,SIG = None,None
def getlabel(f):
    xs =  f.split('/')[-1].split('_')
    #return xs[5] + '_' + xs[-1] + '_' + xs[-2] + '_'  + xs[-3]
    if "infer2" in f:
      lbl = "ASCS[u:(pct:"+xs[-2]+")\u03B1:("+str(float(xs[8].replace("ALPHA","")))+")"
    elif "infer" in f:
      lbl = "ASCS SIG:"+xs[15]
    else:
      lbl =  "CS"

    for l in LABELS:
      if l < len(xs):
         lbl = lbl + ' ' + xs[l]
    return lbl

#plt.tight_layout()

f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]})
#f.set_size_inches(w=13, h=8)
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
            if PLOT_RATIO:
                assert(False ) #fix the hard coding
                ratio = act_ths / np.concatenate(([1.70853752e-07], act_ths[:-1]))
                a0.plot(lens[1:-1], ratio[1:-1], label='Ratio at ' + str(-lens[1] + lens[0])+'_' + getlabel(FILE))
            else:
                a0.plot(lens[mask], act_ths[mask], label='Actual Signal Values')
            LENS = np.copy(lens)
            SIG = np.copy(act_ths)
            if not PLOT_ALL_SIGNALS:
              once = True
        a1.plot(lens[mask], f1maxs[mask], label=getlabel(FILE))

a0.legend(prop={'size':15})
a1.legend(prop={'size':15})
mask = np.multiply(SIG >= MIN_SIG , SIG <= MAX_SIG)
LENS = LENS[mask]
SIG = SIG[mask]
if len(SIG) > 10:
  idx = np.arange(0, len(LENS), int(np.ceil(len(SIG)/10)))
  #if idx[-1] != len(LENS) -1 :
  #  idx = np.append(idx, -1)
else:
  idx = np.arange(0, len(LENS), 1)
#print(mask)
#print(LENS)
#print(SIG)
#print (idx)
#a1.set_xticks(LENS[idx])
#a1.set_xticklabels([ '{}\n({:.1e})'.format(LENS[i], SIG[i]) for i in idx ] , rotation=45, fontsize=12)
a1.set_xlabel('Top number of actual signals', fontsize=15)
#a0.set_ylabel('Actual Signal', fontsize=15)
#a1.set_ylabel('Max F1', fontsize=15)
a0.grid()
a1.grid()
if SAVEFIG is None:
  plt.show()
else:
  plt.savefig(SAVEFIG)

