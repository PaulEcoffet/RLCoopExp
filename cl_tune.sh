#PBS -N rl_hyperparam
#PBS -l walltime=124:00:00
#PBS -l nodes=1:ppn=24
#PBS -m n
#PBS -d /home/ecoffet/robocoop/RLCoopExp/

python=/home/ecoffet/anaconda3/envs/rlcoop/bin/python

$python main_test.py

# Variable definition

echo "Over"
