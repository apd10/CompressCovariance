OLD=$PWD
mkdir -p $PWD/gisette/record
cd $PWD/gisette/
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2
bzip2 -d gisette_scale.bz2
ln -s gisette_scale.bz2 train.txt
cd $OLD
