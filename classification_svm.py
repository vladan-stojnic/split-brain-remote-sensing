#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 08:56:15 2018

@author: Vladan
"""

import numpy as np
import sklearn.svm
import utility
from sklearn.model_selection import train_test_split
import functools
from sklearn.preprocessing import normalize

args = utility.get_parser().parse_args()

def read_feats(path):
    with open(path, 'rb') as f:
        feat = np.load(f)
        x = feat['arr_0']
        y = feat['arr_1']
    return y, x


y, x_single = read_feats(args.features)
y, x_double = read_feats(args.weights)

#x_single = normalize(x_single)
#x_double = normalize(x_double)

x = np.concatenate((x_single, x_double), axis = 1)

print(np.min(x_single))
print(np.max(x_single))
print(np.min(x_double))
print(np.max(x_double))

x = normalize(x)

with open(args.results, 'w') as f:
    for cc in [1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]:
        res = []
        print ('C = {:.4f}'.format(cc))
        f.write ('C = {:.4f}'.format(cc))
        for state in [12, 22, 32, 42, 52, 62]:
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = 0.5, random_state = state)
            
            clf = sklearn.svm.SVC(C = cc, kernel = 'linear')
            clf.fit(x_train, y_train)
            ypred = clf.predict(x_test)
            
            acc = 100.0 * np.mean(ypred == y_test)
            print('Classification accuracy: {:.2f}%'.format(acc))
            f.write('Classification accuracy: {:.2f}%\n'.format(acc))
            res.append(acc)
        avg = functools.reduce(lambda x, y: x+y, res)
        avg /= len(res)
        print('Average classification accuracy: {:.2f}%'.format(avg))
        f.write('Average classification accuracy: {:.2f}%\n'.format(avg))
