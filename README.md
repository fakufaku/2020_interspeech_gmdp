# Generalized Minimal Distortion Principle for Blind Source Separation

## Abstract

We revisit the source image estimation problem from blind source separation
(BSS). We generalize the traditional minimum distortion principle to maximum
likelihood estimation with a model for the residual spectrograms. Because
residual spectrograms typically contain other sources, we propose to use a
mixed-norm model that lets us finely tune sparsity in time and frequency. We
propose to carry out the minimization of the mixed-norm via
majorization-maximization optimization, leading to an iteratively reweighted
least-squares algorithm. The algorithm balances well efficiency and ease of
implementation. We assess the performance of the proposed method as applied
to two well-known determined BSS and one joint BSS-dereverberation algorithms.
We find out that it is possible to tune the parameters to improve separation by
up to 2 dB, with no increase in distortion, and at little computational cost.
The method thus provides a cheap and easy way to boost the performance of
blind source separation.

## Author

[Robin Scheibler](http://robinscheibler.org) (firstname.name@gmail.org)

## Run the experiments

The easiest is to rely on [anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) for the installation.

We use [ipyparallel](https://ipyparallel.readthedocs.io/en/latest/) to parallelize the experiment.

### Setup

```bash
# prepare the environment
git clone --recursive https://git.linecorp.com/robin-scheibler/generalized_minimal_distortion_principle.git
cd generalized_minimal_distortion_principle
conda env create -f environment.yml
conda activate gmdp

# generate the dataset
cd bss_speech_dataset
python ../config_dataset.json
cd ..
```

### Run

```bash
# start the engines
ipcluster start --daemonize

# run experiment for AuxIVA and ILRMA
python ./paper_simulation.py ./experiment1_config.json

# run experiment for ILRMA-T
python ./paper_simulation.py ./experiment1_config.json

# stop the engines
ipcluster stop
```