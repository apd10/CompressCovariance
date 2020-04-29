# Compress covariance matrix

## Setup The data files: (example RCV1)
In the directory containing evaluate.py
```
mkdir -p rcv1/record
cd rcv1
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary
ln -s rcv1.binary train.txt
```

## Run the Evaluation (Sketching & comparing with actual covariances)
```
python3 evaluate.py --dataset rcv1 --insertall --verifyall --countsketch.repetitions 5 --countsketch.range 25000 --insertall.skip_samples_for_mu 100 --use_first_K_features 1000
```

## Visualize the output 
```
python3 view_result.py --file ./rcv1/record/data_INSTrue_VERTrue_CRG25000_CRP5_MUSMP100_NUMFEAT1000_BT1000
```




