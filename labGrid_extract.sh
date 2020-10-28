#!/bin/bash

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABGrid --output-type single -w ./models/LABGrid_single_2_final.hdf5 -f ./features/LABGrid_features_single_2.npy --single-clusters 2 --double-clusters 4
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABGrid --output-type double -w ./models/LABGrid_double_4_final.hdf5 -f ./features/LABGrid_features_double_4.npy --single-clusters 2 --double-clusters 4

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABGrid --output-type single -w ./models/LABGrid_single_8_final.hdf5 -f ./features/LABGrid_features_single_8.npy --single-clusters 8 --double-clusters 16
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABGrid --output-type double -w ./models/LABGrid_double_16_final.hdf5 -f ./features/LABGrid_features_double_16.npy --single-clusters 8 --double-clusters 16

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABGrid --output-type single -w ./models/LABGrid_single_32_final.hdf5 -f ./features/LABGrid_features_single_32.npy --single-clusters 32 --double-clusters 64
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABGrid --output-type double -w ./models/LABGrid_double_64_final.hdf5 -f ./features/LABGrid_features_double_64.npy --single-clusters 32 --double-clusters 64

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABGrid --output-type single -w ./models/LABGrid_single_128_final.hdf5 -f ./features/LABGrid_features_single_128.npy --single-clusters 128 --double-clusters 256
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABGrid --output-type double -w ./models/LABGrid_double_256_final.hdf5 -f ./features/LABGrid_features_double_256.npy --single-clusters 128 --double-clusters 256
