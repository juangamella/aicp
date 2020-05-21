#!/bin/bash -l

# 9 jobs in the finite regime with 300 graphs at different intervention strengths (ridiculously large interventions)

module load new gcc/4.8.2 python/3.7.1

bsub -J "n10i1" -n 50 -W 72:00 python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --i_mean 50 --i_var 5 --batch_size 20000 --max_iter 50 --alpha 0.0002 --finite --n 10 --random_state 0 --tag may14i1
bsub -J "n100i1" -n 50 -W 72:00 python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --i_mean 50 --i_var 5 --batch_size 20000 --max_iter 50 --alpha 0.0002 --finite --n 100 --random_state 0 --tag may14i1
bsub -J "n1000i1" -n 100 -W 120:00 python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --i_mean 50 --i_var 5 --batch_size 20000 --max_iter 50 --alpha 0.0002 --finite --n 1000 --random_state 0 --tag may14i1

bsub -J "n10i2" -n 50 -W 72:00 python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --i_mean 100 --i_var 5 --batch_size 20000 --max_iter 50 --alpha 0.0002 --finite --n 10 --random_state 0 --tag may14i2
bsub -J "n100i2" -n 50 -W 72:00 python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --i_mean 100 --i_var 5 --batch_size 20000 --max_iter 50 --alpha 0.0002 --finite --n 100 --random_state 0 --tag may14i2
bsub -J "n1000i2" -n 100 -W 120:00 python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --i_mean 100 --i_var 5 --batch_size 20000 --max_iter 50 --alpha 0.0002 --finite --n 1000 --random_state 0 --tag may14i2

bsub -J "n10i3" -n 50 -W 72:00 python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --i_mean 200 --i_var 5 --batch_size 20000 --max_iter 50 --alpha 0.0002 --finite --n 10 --random_state 0 --tag may14i3
bsub -J "n100i3" -n 50 -W 72:00 python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --i_mean 200 --i_var 5 --batch_size 20000 --max_iter 50 --alpha 0.0002 --finite --n 100 --random_state 0 --tag may14i3
bsub -J "n1000i3" -n 100 -W 120:00 python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --i_mean 200 --i_var 5 --batch_size 20000 --max_iter 50 --alpha 0.0002 --finite --n 1000 --random_state 0 --tag may14i3