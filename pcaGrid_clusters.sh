#!/bin/bash

set -e

python grid_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --single-clusters 2 --double-clusters 4

python grid_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --single-clusters 8 --double-clusters 16

python grid_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --single-clusters 32 --double-clusters 64

python grid_calculation.py -d /home/research/vladan/data/NWPU-RESISC45 --single-clusters 128 --double-clusters 256
