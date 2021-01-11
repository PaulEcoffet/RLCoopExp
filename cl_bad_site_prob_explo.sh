#PBS -N rl_badsiteprob_gamma0.999
#PBS -l walltime=167:00:00
#PBS -l nodes=1:ppn=24
#PBS -m n
#PBS -t 1-4
#PBS -d /home/ecoffet/robocoop/RLCoopExp/

python=/home/ecoffet/anaconda3/envs/rlcoop/bin/python

TEST=(1 0.5 0.2 0.1)

ID=$((PBS_ARRAYID-1))

$python bad_site_prob_explo.py --gamma 0.999 "${TEST[$ID]}"

# Variable definition

echo "Over"
