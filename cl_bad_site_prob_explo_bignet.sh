#PBS -N rl_badsiteprob_bignet
#PBS -l walltime=167:00:00
#PBS -l nodes=1:ppn=24
#PBS -m n
#PBS -t 1-4
#PBS -q qprio
#PBS -d /home/ecoffet/robocoop/RLCoopExp/

python=/home/ecoffet/anaconda3/envs/rlcoop/bin/python

TEST=(1.0 0.5 0.2 0.1)

ID=$((PBS_ARRAYID-1))

$python bad_site_prob_explo.py --num-per-layer 256 --num-layers 2 "${TEST[$ID]}" --subdir bignetfastredosec

# Variable definition

echo "Over"
