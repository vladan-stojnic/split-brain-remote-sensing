#!/bin/bash

#python pca_decomposition.py -d /home/research/vladan/data/NWPU-RESISC45 --num-pixels 256
#./pcaKmeans_clusters.sh
./pcaKmeans_train.sh
./pcaKmeans_extract.sh
./pcaKmeans_classify.sh
