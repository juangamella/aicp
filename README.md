# Towards active ICP: experiment selection through stability

This repository contains the code to run the experiments and plot the results. This README is not intended to be completely self-explanatory, and should be read alongside the [manuscript](semester_project_juan_gamella.pdf).

## Running the experiments

The experiments can be run with different settings. To run the experiments, execute the following in a terminal:

```
python -m src.run_experiments
```

## Command-line arguments

The experimental settings are controlled via command line arguments. The following give general control over how the experiments are run:

- `n_workers`: Size of the process pool on which to run experiments in parallel. Setting it to `-1` uses as many workers as cores are visible to the parent python process. Default is `1`, ie. running experiments sequentially.
- `batch_size`: Size of the experiment batches which are submitted to the worker pool. A lower batch size reduces the maximum size of allocated memory (useful if running experiments with large graphs, see [this post](https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap)), but increases overhead. Defaults to `20000`.
- `runs`: Number of runs performed over each test case, each initialized with different random seeds. Default is `1`.
- `max_iter`: Maximum number of iterations for which to run each experiment, i.e. maximum number of interventions performed by each policy. Defaults to `-1`, which sets this value to `n_max` (see below).
- `tag`: User-defined label appended to the filename. It has no effect on the experiments, and is disabled by default.
- `random_state`: Sets the random seed, to allow for reproducibility. Consecutive calls with the same parameters should yield the same results. Defaults to `42`.
- `debug`: If the experiments should output debug messages.
- `save_dataset`: Will set the generated test cases into disk, following the directory structure that the ABCD-strategy ([Agrawal et. al 2018](https://arxiv.org/abs/1902.10347)) implementation can read.
- `load_dataset`: Instead of generating new test cases, loads the dataset from the given directory.
- `abcd`: If present, only the Random, e+r and markov+e+r policies are run, for the comparison with ABCD.

Others control the generation of SCMs, i.e. test cases.

- `G`: Number of randomly-generated test cases. Default is `4`.
- `k`: Average degree of the graphs underlying the SEMs. Default is `3`.
- `n_min`: Lower bound for the number of predictor variables, sampled uniformly at random. Defaults to `8`.
- `n_max`: Upper bound for the number of predictor variables, sampled uniformly at random. Defaults to `8`.
- `w_min`: Lower bound for the weights, sampled uniformly at random. Defaults to `0.1`.
- `w_max`: Upper bound for the weights, sampled uniformly at random. Defaults to `1`.
- `var_min`: Lower bound for the variances of the noise variables, sampled uniformly at random. Defaults to `0`.
- `var_max`: Upper bound for the variances of the noise variables, sampled uniformly at random. Defaults to `1`.
- `int_min`: Lower bound for the means of the noise variables (intercepts), sampled uniformly at random. Defaults to `0`.
- `int_max`: Upper bound for the means of the noise variables (intercepts), sampled uniformly at random. Defaults to `1`.

The following give control over the interventions:
- `i_mean`: Mean of the shift intervention. Defaults to `10`.
- `i_var`: Variance of the shift intervention, defaults to `1`.
- `ot`: Maximum number of off-target effects (defaults to `0`). If larger than zero, the number of off-target interventions is picked at random (up to `ot`), and the location of the interventions is then also picked at random (excluding the response).

Additional parameters are available for experiments in the *finite regime*.

- `finite`: If present, experiments are performed in the finite regime, with a sample size specified by parameter `n`. Defaults to `False`.
- `n`: Size of the sample collected in each intervention. Only used if `finite` is present. Defaults to `100`.
- `n_obs`: Size of the initial observational sample. If not specified, this is the same as `n`.
- `alpha`: Level at which ICP is run (not adjusted for the number of iterations). Only used if `finite` is present. Defaults to `0.01`.
- `nsp`: If present, disables the use of the speedup introduced in corollary 3.1.

### Example

Command-line arguments are passed by appending `--`. For example,

```
python -m src.semester_project --n_workers -1 --avg_deg 3 --G 30 --runs 32 --n_min 8 --n_max 8 --w_min 0 --w_max 1 --var_min 0.1 --var_max 1 --int_min 0 --int_max 1 --batch_size 20000 --max_iter 50 --random_state 110 --alpha 0.001 --finite --n 1000 --tag exmple
```

### Result storage

The results from running all experiments are pickled and stored in a file. The filename contains a timestamp and all parameters, so it is always possible to know which file contains which experiments. For example, executing the above example would produce the following file:

```
results_1581038775_n_workers:-1_batch_size:20000_debug:False_avg_deg:3.0_G:30_runs:32_n_min:8_n_max:8_w_min:0.0_w_max:1.0_var_min:0.1_var_max:1.0_int_min:0.0_int_max:1.0_random_state:110_finite:True_max_iter:50_n:10_alpha:0.001_tag:exmple.pickle
```

## Plotting results

The plots can be generated with the notebooks [plots_population.ipynb](plots_population.ipynb) and [plots_finite.ipynb](plots_finite.ipynb). Edit the notebook to select the desired result file and execute. The result files used to generate the plots in the report are set by default and shipped in the repo, so the notebooks can be executed directly.

## Feedback

If you need assistance or have feedback, you are more than welcome to write an email to [gajuan@ethz.ch](mailto:gajuan@ethz.ch)!.