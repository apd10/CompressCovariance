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
The visualization for RCV1 looks like the following, 
a)In the subplot above, we plot K as in topK elements of the set on x - axis and on y-axis, we plot the lower bound of the covariance in this set. So value at x=100 means the covariance lower bound for top 100 covariance values in the covariance matrix.
b) In the subplot below, x-axis is the same. We plot Maximum F1 score on the y-axis for this set.
RCV1:
![RCV1](https://github.com/apd10/CompressCovariance/blob/master/images/rcv1.png)

WEBSPAM:
![WEBSPAM](https://github.com/apd10/CompressCovariance/blob/master/images/webspam.png)

We can plot multiple plots (w and w/o filtering algorithm) on the same plot. The better algorithm will have curve shifted upwards.






