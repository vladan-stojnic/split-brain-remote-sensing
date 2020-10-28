#!/bin/bash

#python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm Zhang --output-type single --initializer he_uniform
python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm Zhang --output-type double --initializer he_uniform
