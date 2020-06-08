for DATASET in gisette;
do
for pct in 99 98 95 90 99.5 80 85 75
do
for frac in 0.05
do
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.005  --target_prob1 0.05 --target_prob2 0.15  --threshold_method infer2 --threshold.infer2.sig_pct $pct --threshold.infer2.init_pct 0  --threshold.infer2.inexp_frac $frac "&"
done
done
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.005  --run_base "&" 
done



for DATASET in epsilon;
do
for pct in 99 95 90 85
do
for frac in 0.05
do
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.1  --target_prob1 0.05 --target_prob2 0.2  --threshold_method infer2 --threshold.infer2.sig_pct $pct --threshold.infer2.init_pct 0  --threshold.infer2.inexp_frac $frac "&"
done
done
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.1  --run_base "&" 
done

for DATASET in cifar10;
do
for pct in 99 95 90 85
do
for frac in 0.05
do
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.05  --target_prob1 0.05 --target_prob2 0.2  --threshold_method infer2 --threshold.infer2.sig_pct $pct --threshold.infer2.init_pct 0  --threshold.infer2.inexp_frac $frac "&"
done
done
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.05  --run_base "&" 
done


for DATASET in rcv1 news20 sector;
do
for pct in 99 98 95 90 99.5 85 80 75
do
for frac in 0.05
do
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.005  --target_prob1 0.05 --target_prob2 0.15  --threshold_method infer2 --threshold.infer2.sig_pct $pct --threshold.infer2.init_pct 0  --threshold.infer2.inexp_frac $frac  "&"
done
done
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.005  --run_base  "&" 
done

echo "#Running Filtered"

for DATASET in gisette_shifted;
do
for pct in 99 98 95 90 99.5
do
for frac in 0.05
do
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.005  --target_prob1 0.05 --target_prob2 0.15  --threshold_method infer2 --threshold.infer2.sig_pct $pct --threshold.infer2.init_pct 0  --threshold.infer2.inexp_frac $frac --filter "&"
done
done
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.005  --run_base --filter "&" 
done



for DATASET in epsilon;
do
for pct in 99 95 90 85
do
for frac in 0.05
do
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.1  --target_prob1 0.05 --target_prob2 0.2  --threshold_method infer2 --threshold.infer2.sig_pct $pct --threshold.infer2.init_pct 0  --threshold.infer2.inexp_frac $frac --filter "&"
done
done
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.1  --run_base --filter "&" 
done

for DATASET in cifar10;
do
for pct in 99 95 90 85
do
for frac in 0.05
do
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.05  --target_prob1 0.05 --target_prob2 0.2  --threshold_method infer2 --threshold.infer2.sig_pct $pct --threshold.infer2.init_pct 0  --threshold.infer2.inexp_frac $frac --filter "&"
done
done
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.05  --run_base --filter "&" 
done


for DATASET in rcv1 news20 sector;
do
for pct in 99 98 95 90 99.5 85 80 75
do
for frac in 0.05
do
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.005  --target_prob1 0.05 --target_prob2 0.15  --threshold_method infer2 --threshold.infer2.sig_pct $pct --threshold.infer2.init_pct 0  --threshold.infer2.inexp_frac $frac  --filter "&"
done
done
echo  nohup taskset -c 29-58 python3 evaluate.py --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha 0.005  --run_base  --filter "&" 
done
