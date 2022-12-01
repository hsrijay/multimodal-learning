#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

# notebook = $1
# scriptname="$(basename $notebook .ipynb)".py

# jupyter nbconvert --to --execute ece.ipynb



python3 train.py 