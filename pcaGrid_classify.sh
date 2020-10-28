#!/bin/bash

python classification_svm.py -w ./features/PCAGrid_features_double_4.npy -f ./features/PCAGrid_features_single_2.npy -r ./results/PCAGrid_2_4.txt
python classification_svm.py -w ./features/PCAGrid_features_double_16.npy -f ./features/PCAGrid_features_single_2.npy -r ./results/PCAGrid_2_16.txt
python classification_svm.py -w ./features/PCAGrid_features_double_64.npy -f ./features/PCAGrid_features_single_2.npy -r ./results/PCAGrid_2_64.txt
python classification_svm.py -w ./features/PCAGrid_features_double_256.npy -f ./features/PCAGrid_features_single_2.npy -r ./results/PCAGrid_2_256.txt

python classification_svm.py -w ./features/PCAGrid_features_double_4.npy -f ./features/PCAGrid_features_single_8.npy -r ./results/PCAGrid_8_4.txt
python classification_svm.py -w ./features/PCAGrid_features_double_16.npy -f ./features/PCAGrid_features_single_8.npy -r ./results/PCAGrid_8_16.txt
python classification_svm.py -w ./features/PCAGrid_features_double_64.npy -f ./features/PCAGrid_features_single_8.npy -r ./results/PCAGrid_8_64.txt
python classification_svm.py -w ./features/PCAGrid_features_double_256.npy -f ./features/PCAGrid_features_single_8.npy -r ./results/PCAGrid_8_256.txt

python classification_svm.py -w ./features/PCAGrid_features_double_4.npy -f ./features/PCAGrid_features_single_32.npy -r ./results/PCAGrid_32_4.txt
python classification_svm.py -w ./features/PCAGrid_features_double_16.npy -f ./features/PCAGrid_features_single_32.npy -r ./results/PCAGrid_32_16.txt
python classification_svm.py -w ./features/PCAGrid_features_double_64.npy -f ./features/PCAGrid_features_single_32.npy -r ./results/PCAGrid_32_64.txt
python classification_svm.py -w ./features/PCAGrid_features_double_256.npy -f ./features/PCAGrid_features_single_32.npy -r ./results/PCAGrid_32_256.txt

python classification_svm.py -w ./features/PCAGrid_features_double_4.npy -f ./features/PCAGrid_features_single_128.npy -r ./results/PCAGrid_128_4.txt
python classification_svm.py -w ./features/PCAGrid_features_double_16.npy -f ./features/PCAGrid_features_single_128.npy -r ./results/PCAGrid_128_16.txt
python classification_svm.py -w ./features/PCAGrid_features_double_64.npy -f ./features/PCAGrid_features_single_128.npy -r ./results/PCAGrid_128_64.txt
python classification_svm.py -w ./features/PCAGrid_features_double_256.npy -f ./features/PCAGrid_features_single_128.npy -r ./results/PCAGrid_128_256.txt
