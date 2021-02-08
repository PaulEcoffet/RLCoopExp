#PBS -N rl_badsiteprob_bignet
#PBS -l walltime=167:00:00
#PBS -l nodes=1:ppn=24
#PBS -m n
#PBS -q qprio
#PBS -d /home/ecoffet/robocoop/RLCoopExp/

python=/home/ecoffet/anaconda3/envs/rlcoop/bin/python

$python time_bench.py --num-per-layer 3 --num-layers 1 --subdir timebench --restore "./logs/paperrun2/e200000/ppobiglr/goodsiteprob_20201126-165448/PPO_partner_choice_b60b6_00023_23_good_site_prob=1.0,max_it=100.0_2020-11-26_20-00-05/checkpoint_583/checkpoint-583" 1.0

$python time_bench.py --num-per-layer 3 --num-layers 1 --subdir timebench --restore "./logs/paperrun2/e200000/ppobiglr/goodsiteprob_20201126-165448/PPO_partner_choice_b613b_00023_23_good_site_prob=0.5,max_it=200.0_2020-11-26_21-09-19/checkpoint_887/checkpoint-887" 0.5

$python time_bench.py --num-per-layer 3 --num-layers 1 --subdir timebench --restore "./logs/paperrun2/e200000/ppobiglr/goodsiteprob_20201126-165448/PPO_partner_choice_b613b_00015_15_good_site_prob=0.5,max_it=200.0_2020-11-26_18-06-45/checkpoint_876/checkpoint-876" 0.2

$python time_bench.py --num-per-layer 3 --num-layers 1 --subdir timebench --restore "./logs/paperrun2/e200000/ppobiglr/0.1rerun/goodsiteprob_20201203-164305/PPO_partner_choice_3c19e_00019_19_good_site_prob=0.1,max_it=1000.0_2020-12-04_01-14-45/checkpoint_1888/checkpoint-1888" 0.1

# Variable definition

echo "Over"
