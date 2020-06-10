OLD=$PWD # you should be in topK directory to run this
mkdir -p $PWD/../url/
cd $PWD/../url/
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2
bzip2 -d url_combined.bz2
head -n 1677291 url_combined > train.txt # we use 70% data for the model. it was just the artifact of our setup. We recommend using same to reproduce the results
cd $OLD
