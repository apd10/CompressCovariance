
for ds in news20.binary url_comp
do 
for RANGE in 1000000 5000000 10000000 #100000000
do
echo nohup python3  training.py --dataset $ds --batch 1 --device_id 1 --countsketch.repetitions 5  --countsketch.range $RANGE --insert.samples_for_mu 1000 --threshold_method constant --threshold.const.thold 0 --threshold.const.theta 0 --threshold.const.exp_frac 0.2  --insert correlation --run_base; 
done
done


for ds in news20.binary
do 
for RANGE in 100000000
do
for theta in 1.2 1.4 1.6 1.8 2.0
do
echo nohup python3  training.py --dataset $ds --batch 1 --device_id 1 --countsketch.repetitions 5  --countsketch.range $RANGE --insert.samples_for_mu 1000 --threshold_method constant --threshold.const.thold 0.0001 --threshold.const.theta $theta --threshold.const.exp_frac 0.2  --insert correlation;  
done

done
done


for ds in news20.binary
do 
for RANGE in 10000000
do
for theta in 3 4 4.5 5 6
do
echo nohup python3  training.py --dataset $ds --batch 1 --device_id 1 --countsketch.repetitions 5  --countsketch.range $RANGE --insert.samples_for_mu 1000 --threshold_method constant --threshold.const.thold 0.0001 --threshold.const.theta $theta --threshold.const.exp_frac 0.2  --insert correlation;  
done

done
done


for ds in news20.binary
do 
for RANGE in 1000000
do
for theta in 10 12 14 16 18
do
echo nohup python3  training.py --dataset $ds --batch 1 --device_id 1 --countsketch.repetitions 5  --countsketch.range $RANGE --insert.samples_for_mu 1000 --threshold_method constant --threshold.const.thold 0.0001 --threshold.const.theta $theta --threshold.const.exp_frac 0.2  --insert correlation;  
done

done
done



for ds in news20.binary
do 
for RANGE in 5000000
do
for theta in  2 4 6 8 10
do
echo nohup python3  training.py --dataset $ds --batch 1 --device_id 1 --countsketch.repetitions 5  --countsketch.range $RANGE --insert.samples_for_mu 1000 --threshold_method constant --threshold.const.thold 0.0001 --threshold.const.theta $theta --threshold.const.exp_frac 0.2  --insert correlation;  
done

done
done



for ds in url_comp
do 
for RANGE in 1000000
do
for theta in 1.3 1.5 1.7 1.9 2.1
do
echo nohup python3  training.py --dataset $ds --batch 1 --device_id 1 --countsketch.repetitions 5  --countsketch.range $RANGE --insert.samples_for_mu 1000 --threshold_method constant --threshold.const.thold 0.0001 --threshold.const.theta $theta --threshold.const.exp_frac 0.2  --insert correlation;  
done
done
done


for ds in url_comp
do 
for RANGE in 10000000
do
for theta in 0.8 0.9 1.0 1.1 1.2
do
echo nohup python3  training.py --dataset $ds --batch 1 --device_id 2 --countsketch.repetitions 5  --countsketch.range $RANGE --insert.samples_for_mu 1000 --threshold_method constant --threshold.const.thold 0.0001 --threshold.const.theta $theta --threshold.const.exp_frac 0.2  --insert correlation;  
done
done
done


for ds in url_comp
do 
for RANGE in 5000000
do
for theta in 0.9 1.0 1.1 1.2 1.3
do
echo nohup python3  training.py --dataset $ds --batch 1 --device_id 2 --countsketch.repetitions 5  --countsketch.range $RANGE --insert.samples_for_mu 1000 --threshold_method constant --threshold.const.thold 0.0001 --threshold.const.theta $theta --threshold.const.exp_frac 0.2  --insert correlation;  
done
done
done

#(base) apd10@yogi:~/experiments/projects/CompressCovariance/topk$ python3 eval_pickle.py  --dataset X --pickle ./record/DSurl_comp_K5_R5000000_TOP1000TH0.0e+00_THETA0.0e+00_EXP0_topK.pickle 
#2.0586846 2.827805 2.1916575
#(base) apd10@yogi:~/experiments/projects/CompressCovariance/topk$ python3 eval_pickle.py  --dataset X --pickle ./record/DSurl_comp_K5_R1000000_TOP1000TH0.0e+00_THETA0.0e+00_EXP0_topK.pickle 
#3.247042 4.561938 3.4455805
#(base) apd10@yogi:~/experiments/projects/CompressCovariance/topk$ python3 eval_pickle.py  --dataset X --pickle ./record/DSurl_comp_K5_R10000000_TOP1000TH0.0e+00_THETA0.0e+00_EXP0_topK.pickle 
#1.8442725 2.5665147 1.9628174

