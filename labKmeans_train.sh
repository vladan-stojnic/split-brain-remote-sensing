#!/bin/bash

set -e

python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type single --single-clusters 2 --double-clusters 16 --initializer he_uniform
python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type double --single-clusters 2 --double-clusters 16 --initializer he_uniform

python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type single --single-clusters 8 --double-clusters 32 --initializer he_uniform
python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type double --single-clusters 8 --double-clusters 32 --initializer he_uniform

python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type single --single-clusters 16 --double-clusters 64 --initializer he_uniform
python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type double --single-clusters 16 --double-clusters 64 --initializer he_uniform

#python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type single --single-clusters 32 --double-clusters 128 --initializer he_uniform
#python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type double --single-clusters 32 --double-clusters 128 --initializer he_uniform

#python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type single --single-clusters 64 --double-clusters 256 --initializer he_uniform
#python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type double --single-clusters 64 --double-clusters 256 --initializer he_uniform

#python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type single --single-clusters 128 --double-clusters 512 --initializer he_uniform
#python train.py -d /home/research/vladan/data/NWPU-RESISC45 --batch-size 100 --algorithm LABKmeans --output-type double --single-clusters 128 --double-clusters 512 --initializer he_uniform
