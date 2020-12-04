#PBS -N rl_badsiteprob
#PBS -l walltime=148:00:00
#PBS -l nodes=1:ppn=24
#PBS -m n
#PBS -t 1-1
#PBS -d /home/ecoffet/robocoop/RLCoopExp/

python=/home/ecoffet/anaconda3/envs/rlcoop/bin/python

TEST=(0.1)

ID=$((PBS_ARRAYID-1))

$python cma_test.py "${TEST[$ID]}"

# Variable definition

echo "Over"
