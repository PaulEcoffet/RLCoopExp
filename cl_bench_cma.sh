#PBS -N bench_cma
#PBS -l walltime=167:00:00
#PBS -l nodes=1:ppn=24
#PBS -m n
#PBS -q qprio
#PBS -d /home/ecoffet/robocoop/RLCoopExp/

python=/home/ecoffet/anaconda3/envs/rlcoop/bin/python

$python cma_bench.py --restore "/home/ecoffet/robocoop/RLCoopExp/./logs/paperrun2/e200000/cmafixed/goodsiteprob_20201214-235729/train_partner_choice_be044_00002_2_good_site_prob=1.0,max_it=100.0_2020-12-14_23-57-29/checkpoint200000/best.pkl" 1.0

$python cma_bench.py  --restore "/home/ecoffet/robocoop/RLCoopExp/./logs/paperrun2/e200000/cmafixed/goodsiteprob_20201214-235739/train_partner_choice_c3f54_00009_9_good_site_prob=0.5,max_it=200.0_2020-12-14_23-57-40/checkpoint200000/best.pkl" 0.5

$python cma_bench.py  --restore "/home/ecoffet/robocoop/RLCoopExp/./logs/paperrun2/e200000/cmafixed/goodsiteprob_20201214-235727/train_partner_choice_bcd0a_00014_14_good_site_prob=0.2,max_it=500.0_2020-12-14_23-57-28/checkpoint200000/best.pkl" 0.2

$python cma_bench.py --restore "/home/ecoffet/robocoop/RLCoopExp/./logs/paperrun2/e200000/cmafixed/goodsiteprob_20201214-235743/train_partner_choice_c6304_00003_3_good_site_prob=0.1,max_it=1000.0_2020-12-14_23-57-43/checkpoint200000/best.pkl" 0.1

# Variable definition

echo "Over"
