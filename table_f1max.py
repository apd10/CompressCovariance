import numpy as np
import pandas as  pd
import sys
import matplotlib.pyplot as plt
import argparse
import pdb
from scipy.interpolate import interp1d
args = argparse.ArgumentParser()
args.add_argument("--files", type=str, action="store", dest="files", required=True)

results = args.parse_args()
FILES = results.files
dfs = []
for FILE in FILES.split(','):
    a = pd.read_csv(FILE)
    a['frac'] = a.len_actual_ids / a.range_actual
    a['f1'] = 2*a.recall * a.precision / (a.recall + a.precision)
    a = a.dropna()
    f1max = a[['frac', 'f1']].groupby('frac').max()
    f1max = f1max.iloc[((f1max.index * 100).astype(int) + 1)%10 == 0,:]
    f1max.reset_index(inplace=True)
    f1max.rename(columns = {'frac': FILE+'FRAC', 'f1' : FILE+'F1'}, inplace=True)
    f1max = f1max.transpose()
    dfs.append(f1max)
if len(dfs) > 1:
  DF = dfs[0].append(dfs[1:])
else:
  DF = dfs[0]
DF.to_csv("~/experiments/projects/CompressCovariance/aggregate_f1max.csv")

