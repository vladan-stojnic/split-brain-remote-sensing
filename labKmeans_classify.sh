#!/bin/bash

python classification_svm.py -w ./features/LABKmeans_features_double_16.npy -f ./features/LABKmeans_features_single_2.npy -r ./results/LABKmeans_2_16.txt
python classification_svm.py -w ./features/LABKmeans_features_double_32.npy -f ./features/LABKmeans_features_single_2.npy -r ./results/LABKmeans_2_32.txt
python classification_svm.py -w ./features/LABKmeans_features_double_64.npy -f ./features/LABKmeans_features_single_2.npy -r ./results/LABKmeans_2_64.txt
python classification_svm.py -w ./features/LABKmeans_features_double_128.npy -f ./features/LABKmeans_features_single_2.npy -r ./results/LABKmeans_2_128.txt
python classification_svm.py -w ./features/LABKmeans_features_double_256.npy -f ./features/LABKmeans_features_single_2.npy -r ./results/LABKmeans_2_256.txt
python classification_svm.py -w ./features/LABKmeans_features_double_512.npy -f ./features/LABKmeans_features_single_2.npy -r ./results/LABKmeans_2_512.txt

python classification_svm.py -w ./features/LABKmeans_features_double_16.npy -f ./features/LABKmeans_features_single_8.npy -r ./results/LABKmeans_8_16.txt
python classification_svm.py -w ./features/LABKmeans_features_double_32.npy -f ./features/LABKmeans_features_single_8.npy -r ./results/LABKmeans_8_32.txt
python classification_svm.py -w ./features/LABKmeans_features_double_64.npy -f ./features/LABKmeans_features_single_8.npy -r ./results/LABKmeans_8_64.txt
python classification_svm.py -w ./features/LABKmeans_features_double_128.npy -f ./features/LABKmeans_features_single_8.npy -r ./results/LABKmeans_8_128.txt
python classification_svm.py -w ./features/LABKmeans_features_double_256.npy -f ./features/LABKmeans_features_single_8.npy -r ./results/LABKmeans_8_256.txt
python classification_svm.py -w ./features/LABKmeans_features_double_512.npy -f ./features/LABKmeans_features_single_8.npy -r ./results/LABKmeans_8_512.txt

python classification_svm.py -w ./features/LABKmeans_features_double_16.npy -f ./features/LABKmeans_features_single_16.npy -r ./results/LABKmeans_16_16.txt
python classification_svm.py -w ./features/LABKmeans_features_double_32.npy -f ./features/LABKmeans_features_single_16.npy -r ./results/LABKmeans_16_32.txt
python classification_svm.py -w ./features/LABKmeans_features_double_64.npy -f ./features/LABKmeans_features_single_16.npy -r ./results/LABKmeans_16_64.txt
python classification_svm.py -w ./features/LABKmeans_features_double_128.npy -f ./features/LABKmeans_features_single_16.npy -r ./results/LABKmeans_16_128.txt
python classification_svm.py -w ./features/LABKmeans_features_double_256.npy -f ./features/LABKmeans_features_single_16.npy -r ./results/LABKmeans_16_256.txt
python classification_svm.py -w ./features/LABKmeans_features_double_512.npy -f ./features/LABKmeans_features_single_16.npy -r ./results/LABKmeans_16_512.txt

python classification_svm.py -w ./features/LABKmeans_features_double_16.npy -f ./features/LABKmeans_features_single_32.npy -r ./results/LABKmeans_32_16.txt
python classification_svm.py -w ./features/LABKmeans_features_double_32.npy -f ./features/LABKmeans_features_single_32.npy -r ./results/LABKmeans_32_32.txt
python classification_svm.py -w ./features/LABKmeans_features_double_64.npy -f ./features/LABKmeans_features_single_32.npy -r ./results/LABKmeans_32_64.txt
#python classification_svm.py -w ./features/LABKmeans_features_double_128.npy -f ./features/LABKmeans_features_single_32.npy -r ./results/LABKmeans_32_128.txt
#python classification_svm.py -w ./features/LABKmeans_features_double_256.npy -f ./features/LABKmeans_features_single_32.npy -r ./results/LABKmeans_32_256.txt
#python classification_svm.py -w ./features/LABKmeans_features_double_512.npy -f ./features/LABKmeans_features_single_32.npy -r ./results/LABKmeans_32_512.txt

python classification_svm.py -w ./features/LABKmeans_features_double_16.npy -f ./features/LABKmeans_features_single_64.npy -r ./results/LABKmeans_64_16.txt
python classification_svm.py -w ./features/LABKmeans_features_double_32.npy -f ./features/LABKmeans_features_single_64.npy -r ./results/LABKmeans_64_32.txt
python classification_svm.py -w ./features/LABKmeans_features_double_64.npy -f ./features/LABKmeans_features_single_64.npy -r ./results/LABKmeans_64_64.txt
#python classification_svm.py -w ./features/LABKmeans_features_double_128.npy -f ./features/LABKmeans_features_single_64.npy -r ./results/LABKmeans_64_128.txt
#python classification_svm.py -w ./features/LABKmeans_features_double_256.npy -f ./features/LABKmeans_features_single_64.npy -r ./results/LABKmeans_64_256.txt
#python classification_svm.py -w ./features/LABKmeans_features_double_512.npy -f ./features/LABKmeans_features_single_64.npy -r ./results/LABKmeans_64_512.txt

python classification_svm.py -w ./features/LABKmeans_features_double_16.npy -f ./features/LABKmeans_features_single_128.npy -r ./results/LABKmeans_128_16.txt
python classification_svm.py -w ./features/LABKmeans_features_double_32.npy -f ./features/LABKmeans_features_single_128.npy -r ./results/LABKmeans_128_32.txt
python classification_svm.py -w ./features/LABKmeans_features_double_64.npy -f ./features/LABKmeans_features_single_128.npy -r ./results/LABKmeans_128_64.txt
#python classification_svm.py -w ./features/LABKmeans_features_double_128.npy -f ./features/LABKmeans_features_single_128.npy -r ./results/LABKmeans_128_128.txt
#python classification_svm.py -w ./features/LABKmeans_features_double_256.npy -f ./features/LABKmeans_features_single_128.npy -r ./results/LABKmeans_128_256.txt
#python classification_svm.py -w ./features/LABKmeans_features_double_512.npy -f ./features/LABKmeans_features_single_128.npy -r ./results/LABKmeans_128_512.txt
