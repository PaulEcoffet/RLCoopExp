#PBS -N rl_badsiteprob_nocritic
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=24
#PBS -m n
#PBS -t 1-2
#PBS -d /home/ecoffet/robocoop/RLCoopExp/

python=/home/ecoffet/anaconda3/envs/rlcoop/bin/python

TEST=(1 0.5)

ID=$((PBS_ARRAYID-1))

$python bad_site_prob_explo.py --no-critic "${TEST[$ID]}"

# Variable definition

echo "Over"
