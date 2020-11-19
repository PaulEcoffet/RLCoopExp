#PBS -N rl_badsiteprob
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=24
#PBS -m n
#PBS -t 1-6
#PBS -d /home/ecoffet/robocoop/RLCoopExp/

python=/home/ecoffet/anaconda3/envs/rlcoop/bin/python

TEST=(1 0.5 0.3 0.2 0.1 0.01)

ID=$((PBS_ARRAYID-1))

$python bad_site_prob_explo.py "${TEST[$ID]}"

# Variable definition

echo "Over"