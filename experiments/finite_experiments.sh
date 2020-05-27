#!/bin/bash -l

# Copyright 2020 Juan Luis Gamella Martin

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# ----------------------------------------------------------------------------
# Commands to run the finite-sample setting experiments presented in the paper
# 9 total: 3 sample sizes (10,100,1000) x 3 levels (0.01, 0.05, 0.1)

python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --max_iter 50 --alpha 0.0002 --finite --n 10 --random_state 0 --tag finl
python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --max_iter 50 --alpha 0.0002 --finite --n 100 --random_state 0 --tag finl
python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --max_iter 50 --alpha 0.0002 --finite --n 1000 --random_state 0 --tag finl

python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --max_iter 50 --alpha 0.0001 --finite --n 10 --random_state 0 --tag finm
python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --max_iter 50 --alpha 0.0001 --finite --n 100 --random_state 0 --tag finm
python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --max_iter 50 --alpha 0.0001 --finite --n 1000 --random_state 0 --tag finm

python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --max_iter 50 --alpha 0.002 --finite --n 10 --random_state 0 --tag finh
python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --max_iter 50 --alpha 0.002 --finite --n 100 --random_state 0 --tag finh
python -m src.run_experiments --n_workers -1 --k 3 --G 300 --runs 8 --p_min 12 --p_max 12 --w_min 0.5 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --max_iter 50 --alpha 0.002 --finite --n 1000 --random_state 0 --tag finh
