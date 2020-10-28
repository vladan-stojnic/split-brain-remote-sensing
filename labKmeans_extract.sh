#!/bin/bash

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type single -w ./models/LABKmeans_single_2_final.hdf5 -f ./features/LABKmeans_features_single_2.npy --single-clusters 2 --double-clusters 16
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type double -w ./models/LABKmeans_double_16_final.hdf5 -f ./features/LABKmeans_features_double_16.npy --single-clusters 2 --double-clusters 16

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type single -w ./models/LABKmeans_single_8_final.hdf5 -f ./features/LABKmeans_features_single_8.npy --single-clusters 8 --double-clusters 32
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type double -w ./models/LABKmeans_double_32_final.hdf5 -f ./features/LABKmeans_features_double_32.npy --single-clusters 8 --double-clusters 32

python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type single -w ./models/LABKmeans_single_16_final.hdf5 -f ./features/LABKmeans_features_single_16.npy --single-clusters 16 --double-clusters 64
python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type double -w ./models/LABKmeans_double_64_final.hdf5 -f ./features/LABKmeans_features_double_64.npy --single-clusters 16 --double-clusters 64

#python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type single -w ./models/LABKmeans_single_32_final.hdf5 -f ./features/LABKmeans_features_single_32.npy --single-clusters 32 --double-clusters 128
#python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type double -w ./models/LABKmeans_double_128_final.hdf5 -f ./features/LABKmeans_features_double_128.npy --single-clusters 32 --double-clusters 128

#python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type single -w ./models/LABKmeans_single_64_final.hdf5 -f ./features/LABKmeans_features_single_64.npy --single-clusters 64 --double-clusters 256
#python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type double -w ./models/LABKmeans_double_256_final.hdf5 -f ./features/LABKmeans_features_double_256.npy --single-clusters 64 --double-clusters 256

#python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type single -w ./models/LABKmeans_single_128_final.hdf5 -f ./features/LABKmeans_features_single_128.npy --single-clusters 128 --double-clusters 512
#python extract.py -d /home/research/vladan/data/AID --batch-size 100 --algorithm LABKmeans --output-type double -w ./models/LABKmeans_double_512_final.hdf5 -f ./features/LABKmeans_features_double_512.npy --single-clusters 128 --double-clusters 512
