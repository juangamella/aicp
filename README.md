# semester_project

This repository contains the code to run experiments and plot results.

### Running the experiments

The experiments can be run interactively with different settings. To run the experiments, execute the following in a terminal:

```
python -m src.semester_project
```

The parameters for the experiments can be set by passing the following command-line arguments:

-`n_workers`: Size of the process pool on which to run experiments in parallel. `-1` uses as many workers as cores are visible by the python process. Default is `1`, ie. running experiments sequentially.
- `batch_size`: Size of the experiment batches which are submitted to the worker pool. A lower batch size reduces the maximum size of allocated memory (useful if running experiments with large graphs, see (this post)[https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap]), but increases overhead. Defaults to `20000`.
- `runs`: Number of runs performed over each test case, each initialized with different random seeds. Default is `1`.
- `max_iter`: Maximum number of iterations for which to run each experiment, i.e., maximum number of interventions performed by each policy. Defaults to `-1`, which sets this value to `n_max`.


The following parameters control the generation of SEMs, i.e. test cases.

- `G`: Number of randomly-generated test cases. Default is `4`.
- `avg_deg`: Average degree of the graphs underlying the SEMs. Default is `3`.
- `n_min`: Lower bound for the number of predictor variables, sampled uniformly at random. Defaults to `8`.
- `n_max`: Upper bound for the number of predictor variables, sampled uniformly at random. Defaults to `8`.
- `w_min`: Lower bound for the weights, sampled uniformly at random. Defaults to `0.1`.
- `w_max`: Upper bound for the weights, sampled uniformly at random. Defaults to `0.2`.
- `var_min`: Lower bound for the variances of the noise variables, sampled uniformly at random. Defaults to `0.1`.
- `var_max`: Upper bound for the variances of the noise variables, sampled uniformly at random. Defaults to `1`.
- `int_min`: Lower bound for the intercepts, sampled uniformly at random. Defaults to `0`.
- `int_max`: Upper bound for the intercepts, sampled uniformly at random. Defaults to `1`.

Additional parameters are available for experiments in the *finite regime*.

- `finite`: If present, experiments are performed in the finite regime, with a sample size specified by parameter `n`. Defaults to `False`.
- `n`: Size of the sample collected in each intervention. Only used if `finite` is present. Defaults to `100`.
- `alpha`: Level of the tests used in ICP. Only used if `finite` is present. Defaults to `0.01`.
- `random_state`: Random state of the test case generator, to allow for reproducibility. Defaults to `42`.
- `tag`: User-defined label appended to the filename. Has no effect on the experiments, and is disabled by default.

**Example**

Command-line arguments are passed by appending `--`. For example,

```
python -m src.semester_project --n_workers -1 --avg_deg 3 --G 30 --runs 32 --n_min 8 --n_max 8 --w_min 0 --w_max 1 --var_min 0.1 --var_max 1 --int_min 0 --int_max 1 --batch_size 20000 --max_iter 50 --random_state 110 --alpha 0.001 --finite --n 1000 --tag exmple
```

**Result storage**

The results from running all experiments are pickled and stored in a file. The filename contains a timestamp and all parameters, so it is always possible to know which file contains which experiments. For example, executing the above example would produce the following file:

```
results_1581038775_n_workers:-1_batch_size:20000_debug:False_avg_deg:3.0_G:30_runs:32_n_min:8_n_max:8_w_min:0.0_w_max:1.0_var_min:0.1_var_max:1.0_int_min:0.0_int_max:1.0_random_state:110_finite:True_max_iter:50_n:10_alpha:0.001_tag:exmple.pickle```

**Plotting results**

The plots can be generated with the notebooks `plots_population.ipynb` and `plots_finite.ipynb`. Edit the notebook to add the result file and execute. The result files used to generate the plots in the report are set by default and shipped in the repo, so the notebooks can be executed directly.