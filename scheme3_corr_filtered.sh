ALPHA=0.02
PCT=98
DATASET=gisette_shifted
echo  nohup taskset -c 29-58 python3 evaluate.py --filter  --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --eval_alpha $ALPHA  --run_base "&" 
for pct in 95 98 99
do
for frac in 0.05
do
echo  nohup taskset -c 10-58 python3 evaluate.py --filter  --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha $ALPHA  --target_prob1 0.05 --target_prob2 0.15  --threshold_method infer2 --threshold.infer2.sig_pct $pct --threshold.infer2.init_pct 0  --threshold.infer2.inexp_frac $frac --eval_alpha $ALPHA "&"
done
done

ALPHA=0.01
PCT=99
for DATASET in rcv1 sector  ;
do
echo  nohup taskset -c 29-58 python3 evaluate.py --filter  --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --eval_alpha $ALPHA  --run_base "&" 
for pct in 99 98 95
do
for frac in 0.05
do
echo  nohup taskset -c 10-58 python3 evaluate.py --filter  --dataset $DATASET --insert correlation --countsketch.repetitions 5 --countsketch.range 20000 --insert.samples_for_mu 500 --use_first_K_features 1000 --alpha $ALPHA  --target_prob1 0.05 --target_prob2 0.15  --threshold_method infer2 --threshold.infer2.sig_pct $pct --threshold.infer2.init_pct 0  --threshold.infer2.inexp_frac $frac --eval_alpha $ALPHA "&"
done
done
done
