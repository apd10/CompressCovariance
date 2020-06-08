import pandas as pd  
import numpy as np   
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
    d = pd.read_csv(FILE, sep=",", header=None) 
    d.columns = ["len", "frac", "meansig"]
    d = d[["frac", "meansig"]]
    d = d.iloc[[0,2,4,12,25,49]]
    d.rename(columns={"frac": FILE+"FRAC", "meansig" : FILE+"SIG"}, inplace=True)
    dfs.append(d.transpose())
if len(dfs) > 1:
  DF = dfs[0].append(dfs[1:])
else:
  DF = dfs[0]
#print(DF)
DF.to_csv("~/experiments/projects/CompressCovariance/aggregate_sigs.csv")
