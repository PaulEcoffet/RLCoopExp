#PBS -N rl_bench_bignet
#PBS -l walltime=167:00:00
#PBS -l nodes=1:ppn=24
#PBS -m n
#PBS -q qprio
#PBS -d /home/ecoffet/robocoop/RLCoopExp/

python=/home/ecoffet/anaconda3/envs/rlcoop/bin/python

$python time_bench.py --num-per-layer 256 --num-layers 2 --subdir timebenchdeep --restore "logs/paperrun3/bignetfastredo/critic_True+episode_200000+gamma_1+num_layers_2+num_per_layers_256/goodsiteprob_20210121-180354/PPO_partner_choice_a49b4_00007_7_good_site_prob=1.0,max_it=100,gamma=1,lr=0.001_2021-01-21_18-03-54/checkpoint_1613/checkpoint-1613" 1.0

$python time_bench.py --num-per-layer 256 --num-layers 2 --subdir timebenchdeep --restore "logs/paperrun3/bignetfastredo/critic_True+episode_200000+gamma_1+num_layers_2+num_per_layers_256/goodsiteprob_20210121-180355/PPO_partner_choice_a50be_00003_3_good_site_prob=0.5,max_it=200,gamma=1,lr=0.001_2021-01-21_18-03-56/checkpoint_1456/checkpoint-1456" 0.5

$python time_bench.py --num-per-layer 256 --num-layers 2 --subdir timebenchdeep --restore "logs/paperrun3/bignetfastredo/critic_True+episode_200000+gamma_1+num_layers_2+num_per_layers_256/goodsiteprob_20210121-180420/PPO_partner_choice_b425b_00002_2_good_site_prob=0.2,max_it=500,gamma=1,lr=0.001_2021-01-21_18-04-21/checkpoint_2550/checkpoint-2550" 0.2

$python time_bench.py --num-per-layer 256 --num-layers 2 --subdir timebenchdeep --restore "logs/paperrun3/bignetfastredo/critic_True+episode_200000+gamma_1+num_layers_2+num_per_layers_256/goodsiteprob_20210121-180445/PPO_partner_choice_c3315_00007_7_good_site_prob=0.1,max_it=1000,gamma=1,lr=0.001_2021-01-21_18-04-46/checkpoint_8369/checkpoint-8369" 0.1

# Variable definition

echo "Over"
