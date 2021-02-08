# RLCoopExp

## Installation

Install ray, cma, torch, rllib, pandas, seaborn, numpy and jupyter notebooks.

```bash
conda install torch numpy pandas seaborn matplotlib jupyter
pip install dm_tree ray "ray[rllib]" cma
```

## Reproduce simulations

```bash
# ppo mlp
python bad_site_prob_explo.py --subdir "ppo-mlp" 1.0 0.5 0.2 0.1

# cma es
python cma_test.py 1.0 0.5 0.2 0.1

# ppo deep
python bad_site_prob_explo.py --num-layers 2 --num-per-layers 256 --subdir "ppo-deep" 1.0 0.5 0.2 0.1
```

## run analysis

```bash
python evaluate_at_checkpoint.py
```

## Do graphs

Open jupyter notebooks and generates the graph. You might have to correct paths

```bash
jupyter lab
```