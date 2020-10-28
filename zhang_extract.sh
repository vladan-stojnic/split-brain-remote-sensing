#!/bin/bash

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm Zhang --output-type single -w ./models/Zhang_single_100_final.hdf5 -f ./features/Zhang_features_single.npy
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm Zhang --output-type double -w ./models/Zhang_double_313_final.hdf5 -f ./features/Zhang_features_double.npy
