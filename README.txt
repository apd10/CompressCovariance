# Experiment 1: generate the results given in the table1 and figure1
BASE_DIR = directory in which evaluate.py is located

## Setup Datasets: 
By default we have a script for gisette dataset. you can change the name of dataset and the http path to the svm file to generate data for other datasets. To generate the gisette data
Run :
sh  setup_datasets.sh
You can see that the gisette folder is created with the dataset and an empty folder "record". The runs of algorithm will dump values in this directory

## Run the commands.
To generate relevant commands for gisette run the following :  (these run on CPU. we ran it on 28 cores. feel free to change parameters of taskset -c <> to adjust to requirements. 28core takes hardly a minute to run)
bash exp1.sh | grep gisette

Every command line will generate two files in the record folder.
> Files correlations* will store the values required for table 1.
> Files data* will be used to plot figures


# FIGURE 1
cd $BASE_DIR/gisette/record/
to generate table 1 (a)
python3 ../../view_f1max.py --files $(ls data*ALPHA2*.0_* data*constant* | xargs | sed 's/ /,/g') --max_sig 0.95 --min_sig 0.05
to generate table 1 (f)
python3 ../../view_f1max.py --files $(ls data*98.0_* data*constant* | xargs | sed 's/ /,/g') --max_sig 0.95 --min_sig 0.05
#TABLE 1
the correlations* files have topK correlations in csv format K,fraction_of_alphap,mean_correlation


# Experiment 2
# base directory is $BASE_DIR/topK
cd $BASE_DIR/topK/
sh setup_datasets.sh

To generate the results for URL dataset # These require  a GPU to run . We ran it on 16 GB GPU. However this command for url with range 10^6 can run in lower memory < 4GB . If it crashes, just reduce the batch size.

nohup python3 training.py --dataset url --batch 50 --device_id 0 --countsketch.repetitions 5 --countsketch.range 1000000 --insert.samples_for_mu 1000 --threshold_method constant --threshold.const.thold 0.0001 --threshold.const.theta 1.7 --threshold.const.exp_frac 0.2 --insert correlation
# run_base CS
nohup python3 training.py --dataset url --batch 50 --device_id 0 --countsketch.repetitions 5 --countsketch.range 1000000 --insert.samples_for_mu 1000 --threshold_method constant --threshold.const.thold 0.0001 --threshold.const.theta 1.7 --threshold.const.exp_frac 0.2 --insert correlation --run_base # run_base overrides all configs


