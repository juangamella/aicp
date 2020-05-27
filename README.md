# Active Invariant Causal Prediction: Experiment Selection through Stability

This repository contains the code to run the experiments and plot the results. This README is not intended to be completely self-explanatory, and should be read alongside the manuscript.

## Installing dependencies

You will need at least Python 3.6. All dependencies are specified in [`requirements.txt`](requirements.txt). To create a virtual environment and install them, run:

```
virtualenv --no-site-packages venv
. venv/bin/activate
pip install -r requirements.txt
```

To run the notebooks from the virtual environments, create a new local kernel:

```
ipython kernel install --user --name=.venv
```

and once inside the notebook select the kernel: `Kernel -> Change kernel -> .venv`.

*Note: Before running the above, make sure you're not already inside a virtual environment. Run `deactivate` *

## Running experiments

The experiments can be run with different settings. To run the experiments with default settings, execute the following in a terminal:

```
python -m src.run_experiments
```

The exact commands to reproduce the results presented in the paper are given [below](#rep).

### Command-line arguments

The experimental settings are controlled via command line arguments. The following give general control over how the experiments are run:

- `n_workers`: Size of the process pool on which to run experiments in parallel. Setting it to `-1` uses as many workers as cores are visible to the parent python process. Default is `1`, ie. running experiments sequentially.
- `batch_size`: Size of the experiment batches which are submitted to the worker pool. A lower batch size reduces the maximum size of allocated memory (useful if running experiments with large graphs, see [this post](https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap)), but increases overhead. Defaults to `20000`.
- `runs`: Number of runs performed over each test case, each initialized with different random seeds. Default is `1`.
- `max_iter`: Maximum number of iterations for which to run each experiment, i.e. maximum number of interventions performed by each policy. Defaults to `-1`, which sets this value to `p_max` (see below).
- `tag`: User-defined label appended to the filename. It has no effect on the experiments, and is disabled by default.
- `random_state`: Sets the random seed, to allow for reproducibility. Consecutive calls with the same parameters should yield the same results. Defaults to `42`.
- `debug`: If the experiments should output debug messages.
- `save_dataset`: Will set the generated test cases into disk, following the directory structure that the [implementation](https://github.com/juangamella/active_learning) for the ABCD-strategy ([Agrawal et. al 2018](https://arxiv.org/abs/1902.10347)) can read. Note that this prevents the experiments from running, i.e. the program stops after the dataset is generated.
- `load_dataset`: Instead of generating new test cases, loads the dataset from the given directory.
- `abcd`: If present, only the Random, e, r and e+r policies are run, for the comparison with ABCD.

Others control the generation of SCMs, i.e. test cases.

- `G`: Number of randomly-generated test cases. Default is `4`.
- `k`: Average degree of the graphs underlying the SEMs. Default is `3`.
- `p_min`: Lower bound for the number of predictor variables, sampled uniformly at random. Defaults to `8`.
- `p_max`: Upper bound for the number of predictor variables, sampled uniformly at random. Defaults to `8`.
- `w_min`: Lower bound for the weights, sampled uniformly at random. Defaults to `0.1`.
- `w_max`: Upper bound for the weights, sampled uniformly at random. Defaults to `1`.
- `var_min`: Lower bound for the variances of the noise variables, sampled uniformly at random. Defaults to `0`.
- `var_max`: Upper bound for the variances of the noise variables, sampled uniformly at random. Defaults to `1`.
- `int_min`: Lower bound for the means of the noise variables (intercepts), sampled uniformly at random. Defaults to `0`.
- `int_max`: Upper bound for the means of the noise variables (intercepts), sampled uniformly at random. Defaults to `1`.

The following give control over the interventions:
- `do`: If present, the interventions are do interventions. If not, shift interventions are carried out (the default).
- `i_mean`: Mean of the intervention. Defaults to `10`.
- `i_var`: Variance of the intervention, defaults to `1`.
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
python -m src.run_experiments --n_workers -1 --k 3 --G 30 --runs 32 --p_min 8 --p_max 8 --w_min 0 --w_max 1 --var_min 0 --var_max 1 --int_min 0 --int_max 1 --batch_size 20000 --max_iter 50 --random_state 110 --alpha 0.001 --finite --n 1000 --tag exmple
```

### Result storage

The results from running all experiments are pickled and stored in a file. The filename contains a timestamp and all parameters, so it is always possible to know which file contains which experiments. For example, executing the above example would produce the following file:

```
results_1581038775_n_workers:-1_batch_size:20000_debug:False_avg_deg:3.0_G:30_runs:32_p_min:8_p_max:8_w_min:0.0_w_max:1.0_var_min:0_var_max:1.0_int_min:0.0_int_max:1.0_random_state:110_finite:True_max_iter:50_n:10_alpha:0.001_tag:exmple.pickle
```

## Reproducing the results<a name="rep"></a>

The commands to reproduce the results from the paper can be found in the `experiments/` directory. In particular,

- [`experiments/population_experiments.sh`](experiments/population_experiments.sh) for the population setting experiments,
- [`experiments/finite_experiments.sh`](experiments/finite_experiments.sh) for the experiments for the finite regime, and
- [`experiments/intervention_strength_experiments.sh`](experiments/intervention_strength_experiments.sh) for the experiments comparing intervention strengths.

The commands to reproduce the A-ICP vs. ABCD results can be found in [`experiments/abcd_experiments.sh`](experiments/abcd_experiments.sh). These commands generate a dataset and execute A-ICP on it. To execute ABCD, the dataset must be then copied over to the [repository](https://github.com/juangamella/active_learning) that contains the ABCD implementation. Exact instructions on how to run ABCD to reproduce the experiments can be found in that repository.

The evaluation of the Markov-blanket estimation is carried out in the [`mb_estimation_analysis.ipynb`](mb_estimation_analysis.ipynb) notebook.

### Plotting

The plots can be generated with the plotting notebooks, i.e. `plots_*`. Edit the notebook to add the path to the desired result files and execute.

## Feedback

If you need assistance or have feedback, you are more than welcome to write an email to [gajuan@ethz.ch](mailto:gajuan@ethz.ch)!.
