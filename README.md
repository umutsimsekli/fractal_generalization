# jhi_calculator


This code repository includes the source code for the [Paper](https://proceedings.neurips.cc/paper/2021/file/9bdb8b1faffa4b3d41779bb495d79fb9-Paper.pdf):

```
Camuto, A., Deligiannidis, G., Erdogdu, M. A., Gurbuzbalaban, M., Simsekli, U., & Zhu, L. "Fractal structure and generalization properties of stochastic optimization algorithms.", Advances in Neural Information Processing Systems (NeurIPS), 2021.
```

It is used for estimating the dominant eigenvalue of the Jacobian of SGD iterates.


## Set Up on Unix

1. Download conda at https://docs.conda.io/en/latest/
2. run `conda create -n jhi python=3.8.5`
3. run `conda activate jhi` to enter the conda env
4. run `python setup.py install` to install the package
4. to use the tool run `jhi --help`.

## Tool

Using the tool is very simple. Consider the following command:

```
jhi -f /path/to/results/20210407_fcn_cifar10/02048_5_00_fcn_cifar10_NLL_0.0_1.0_50_0.135/extra_iters -b 50 -l 0.135 -d cifar10 -m 5 -g true -d ./results
```

- `-f` path to a folder containing pytorch models saved in the form `model_name.ptY`
- `-b` the batch size
- `-l` the learning rate
- `-d` dataset, one of 'cifar10', 'cifar100', 'mnist'.
- `-m` maximum number of batches to use
- `-g` boolean gpu flag
- `-s` directory to save results to

The command triggers a routine that estimates the dominant eigenvalue of
the Jacobian of SGD iterates for each `.ptY` file in the folder specified by `-f`.
This estimation occurs for `-m` batches of size `-b` for each model.

Results are saved in `-f` as a pickled pandas dataframe `dom_eig.pkl`.
To read these results run `df = pandas.read_pickle(/path/to/folder/dom_eig.pkl)`.

To then calculate the mean spectral norm: `msn = df.abs().get_values().mean()`.

## Bash

Run the bash script to iterate over many folders.
Set the `RESULTS_ROOT_FOLDER` env variable. eg:
```
export RESULTS_ROOT_FOLDER=/path/to/results/
```
Set the `GPU` env variable to true or false. eg:

```
export GPU=true
```

Then run `bash compile_and_run.sh $RESULTS_ROOT_FOLDER $GPU`
