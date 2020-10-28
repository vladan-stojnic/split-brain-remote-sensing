#!/bin/bash

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm Zhang --output-type single -w ./models/Zhang_single_100_weights.15-3.86.hdf5 -f ./features/Zhang_features_single_15.npy
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm Zhang --output-type double -w ./models/Zhang_double_313_weights.15-1.87.hdf5 -f ./features/Zhang_features_double_15.npy

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm Zhang --output-type single -w ./models/Zhang_single_100_weights.30-3.72.hdf5 -f ./features/Zhang_features_single_30.npy
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm Zhang --output-type double -w ./models/Zhang_double_313_weights.30-1.66.hdf5 -f ./features/Zhang_features_double_30.npy

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm Zhang --output-type single -w ./models/Zhang_single_100_weights.45-3.63.hdf5 -f ./features/Zhang_features_single_45.npy
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm Zhang --output-type double -w ./models/Zhang_double_313_weights.45-1.58.hdf5 -f ./features/Zhang_features_double_45.npy

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm Zhang --output-type single -w ./models/Zhang_single_100_weights.60-3.62.hdf5 -f ./features/Zhang_features_single_60.npy
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm Zhang --output-type double -w ./models/Zhang_double_313_weights.60-1.55.hdf5 -f ./features/Zhang_features_double_60.npy

python classification_svm.py -w ./features/Zhang_features_double_15.npy -f ./features/Zhang_features_single_15.npy -r ./results/Zhang_15.txt
python classification_svm.py -w ./features/Zhang_features_double_30.npy -f ./features/Zhang_features_single_30.npy -r ./results/Zhang_30.txt
python classification_svm.py -w ./features/Zhang_features_double_45.npy -f ./features/Zhang_features_single_45.npy -r ./results/Zhang_45.txt
python classification_svm.py -w ./features/Zhang_features_double_60.npy -f ./features/Zhang_features_single_60.npy -r ./results/Zhang_60.txt
