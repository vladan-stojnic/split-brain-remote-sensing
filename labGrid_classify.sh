#!/bin/bash

python classification_svm.py -w ./features/LABGrid_features_double_4.npy -f ./features/LABGrid_features_single_2.npy -r ./results/LABGrid_2_4.txt
python classification_svm.py -w ./features/LABGrid_features_double_16.npy -f ./features/LABGrid_features_single_2.npy -r ./results/LABGrid_2_16.txt
python classification_svm.py -w ./features/LABGrid_features_double_64.npy -f ./features/LABGrid_features_single_2.npy -r ./results/LABGrid_2_64.txt
python classification_svm.py -w ./features/LABGrid_features_double_256.npy -f ./features/LABGrid_features_single_2.npy -r ./results/LABGrid_2_256.txt

python classification_svm.py -w ./features/LABGrid_features_double_4.npy -f ./features/LABGrid_features_single_8.npy -r ./results/LABGrid_8_4.txt
python classification_svm.py -w ./features/LABGrid_features_double_16.npy -f ./features/LABGrid_features_single_8.npy -r ./results/LABGrid_8_16.txt
python classification_svm.py -w ./features/LABGrid_features_double_64.npy -f ./features/LABGrid_features_single_8.npy -r ./results/LABGrid_8_64.txt
python classification_svm.py -w ./features/LABGrid_features_double_256.npy -f ./features/LABGrid_features_single_8.npy -r ./results/LABGrid_8_256.txt

python classification_svm.py -w ./features/LABGrid_features_double_4.npy -f ./features/LABGrid_features_single_32.npy -r ./results/LABGrid_32_4.txt
python classification_svm.py -w ./features/LABGrid_features_double_16.npy -f ./features/LABGrid_features_single_32.npy -r ./results/LABGrid_32_16.txt
python classification_svm.py -w ./features/LABGrid_features_double_64.npy -f ./features/LABGrid_features_single_32.npy -r ./results/LABGrid_32_64.txt
python classification_svm.py -w ./features/LABGrid_features_double_256.npy -f ./features/LABGrid_features_single_32.npy -r ./results/LABGrid_32_256.txt

python classification_svm.py -w ./features/LABGrid_features_double_4.npy -f ./features/LABGrid_features_single_128.npy -r ./results/LABGrid_128_4.txt
python classification_svm.py -w ./features/LABGrid_features_double_16.npy -f ./features/LABGrid_features_single_128.npy -r ./results/LABGrid_128_16.txt
python classification_svm.py -w ./features/LABGrid_features_double_64.npy -f ./features/LABGrid_features_single_128.npy -r ./results/LABGrid_128_64.txt
python classification_svm.py -w ./features/LABGrid_features_double_256.npy -f ./features/LABGrid_features_single_128.npy -r ./results/LABGrid_128_256.txt
