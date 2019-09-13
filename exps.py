#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:53:40 2019

@author: linyuanhang
"""

import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation

def demo(unlabeled_per = 0.5):
    label_prop_model = LabelPropagation(kernel = "rbf", gamma=20, 
                                        n_neighbors=7, max_iter=1000, tol=0.001)
    
    # load data
    iris = datasets.load_iris()
    data, target = iris.data, iris.target
    
    # randomly makes some data unlabeled
    rng = np.random.RandomState(0)
    # uniform distribution over [0, 1)
    rand_unlabeled_idx = rng.rand(len(data)) < unlabeled_per
    
    true_target = target.copy()
    target[rand_unlabeled_idx] = -1
    label_prop_model.fit(data, target)
    predicts = label_prop_model.predict(data[rand_unlabeled_idx])
    total = predicts.shape[0]
    accu = np.where(true_target[rand_unlabeled_idx]==predicts)[0].shape[0]
    print("done, unlabel rate:", unlabeled_per, "total:", total, "accu:", accu, "accuracy:", accu/total)
    

def exp1():
    pass

def exp2():
    pass

if __name__ == "__main__":
    exp1()
    exp2()
    for i in range(1, 10):
        demo(unlabeled_per = i/10)