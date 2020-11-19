#PBS -N rl_badsiteprob
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=24
#PBS -m n
#PBS -d /home/ecoffet/robocoop/RLCoopExp/

python=/home/ecoffet/anaconda3/envs/rlcoop/bin/python

$python bad_site_prob_explo.py

# Variable definition

echo "Over"
