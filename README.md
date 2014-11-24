pylearn2_mnist
==============

training MNIST data with pylearn2

## Preparation

* Install Pylean2  
see http://deeplearning.net/software/pylearn2/
* Set PYLEARN2_VIEWER_COMMAND environment variable

## Download dataset

Pylearn2 have a script to download MNIST dataset.  
Please execute pylearn2/scripts/datasets/download_mnist.py

## Show dataset examples

Execute show_samples.py

## Train

Execute test_dae.py

## Show result

Execute test_result.py to recognize test data and output result.  
result is (Number of correctly recognized data) / (Number of total test data)

## Show error data

Execute show_error_data.py to show data incorrectly recognized.
