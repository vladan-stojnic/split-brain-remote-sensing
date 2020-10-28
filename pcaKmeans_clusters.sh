#!/bin/bash

set -e

#python kmeans_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --num-pixels 500 --single-clusters 2 --double-clusters 2
#python kmeans_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --num-pixels 500 --single-clusters 100 --double-clusters 313

python kmeans_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --num-pixels 500 --single-clusters 4 --double-clusters 16

python kmeans_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --num-pixels 500 --single-clusters 8 --double-clusters 32

python kmeans_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --num-pixels 500 --single-clusters 16 --double-clusters 64

#python kmeans_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --num-pixels 500 --single-clusters 32 --double-clusters 128

#python kmeans_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --num-pixels 500 --single-clusters 64 --double-clusters 256

#python kmeans_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --num-pixels 600 --single-clusters 128 --double-clusters 512
