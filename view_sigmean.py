import pandas as pd  
import numpy as np   
import sys
import matplotlib.pyplot as plt
import argparse
import pdb
from scipy.interpolate import interp1d

args = argparse.ArgumentParser()
args.add_argument("--files", type=str, action="store", dest="files", required=True)
args.add_argument("--savefig", type=str, action="store", dest="savefig", default=None)
args.add_argument("--label", type=str, action="store", dest="label", default="-2")


results = args.parse_args()
FILES = results.files
SAVEFIG = results.savefig

LABELS = [int(i) for i in results.label.split(',')]
LENS,SIG = None,None
def getlabel(f):
    xs =  f.split('/')[-1].split('_')
    #return xs[5] + '_' + xs[-1] + '_' + xs[-2] + '_'  + xs[-3]
    lbl = xs[LABELS[0]]
    for l in LABELS[1:]:
      if l < len(xs):
         lbl = lbl + '_' + xs[l]
    return lbl.replace("EXP0_0", "BASE")

for FILE in FILES.split(','):
    print(FILE)
    d = pd.read_csv(FILE, sep=",", header=None) 
    d.columns = ["len","frac","meansig"]
    plt.plot(d.frac, d.meansig, label=getlabel(FILE))

plt.legend()
plt.xlabel('Top reported by Sketch')
plt.ylabel('Mean Signal')
if SAVEFIG is None:
  plt.show()
else:
  plt.savefig(SAVEFIG)

