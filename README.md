# Project Description

<https://github.com/QiuRunwen/CategoryEncoderComparison>

This is python source code/data files for our paper ''Comparative Study on the Performance of Categorical Variable Encoders in Classification and Regression Tasks''.

## Files

```text
data/ # Datasets downloaded from various repositories
exp_conf/ # The configuration of the experiments on natural datasets
Library/ # Files used to fix existing package bugs
output/ # The output of the source code after running
    data_desc/ # The details of the datasets. Output by `src/datasets.py`
    result/ # The result on natural datasets. Output by `src/exp.py`
        current/
            result.csv # The results on natural datasets.
            conf.json # The configuration the experiments.
src/
    data/ # Prepare datasets.
        util.py # Common functions
        *.py # One file per data set.
    analysis_result.py # Analyze the result on natural datasets
    analysis_synthetic.py # Analyze the result on synthetic datasets
    datasets.py # load and describe the datasets.
    *_encoder.py # The encoders implemented by us.
    exp.py # The experiments on natural datasets
    exp_conf.py # The generation of the configuration.
    models.py # The classifiers and regressors.
    preprocess.py # The preprocessing of the datasets including encoding.
    utils.py # Common functions.
tests/
    test_*.py # test the core functions in `src/*.py`
```

## Environment

Step1: install python and related packages.

```bash
conda create -n cat_enc python=3.11.3
conda activate cat_enc
pip install -r requirement.txt
```

Step2: Fix some error in current packages. see [readme.txt](Library/readme.txt)

## Run experiments

```bash
cd src
python exp.py
```

About a month, using an i7-7920HQ CPU @ 3.10GHz and 32GB RAM without GPU acceleration.

## Run analyses

```bash
cd src
python datasets.py
python analysis_result.py
python analysis_synthetic.py
```
