#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:53:40 2019

@author: linyuanhang
"""

import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
import matplotlib.pyplot as plt

def demo_label_propgation_2002paper_iris_dataset(unlabeled_per = 0.5):
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
    

def demo_label_propgation_2002paper_same_distribution(unlabeled_percentage_ = 0.1, seed_ = 0):
    
    mu_1 = np.array([2, 2])
    sigma_1 = 0.01 * np.diag(np.ones(2))
    data_class_1 = np.random.multivariate_normal(mu_1, sigma_1, 100)
    labels_class_1 = np.ones(100).astype(int)
    
    mu_2 = np.array([-2, -2])
    sigma_2 = 0.1 * np.diag(np.ones(2))
    data_class_2 = np.random.multivariate_normal(mu_2, sigma_2, 100)
    labels_class_2 = np.zeros(100).astype(int)
    
    data = np.vstack((data_class_1, data_class_2))
    labels = np.concatenate((labels_class_1, labels_class_2))
    
    plt.scatter(data[0:100,0], data[0:100,1], c = 'red')
    plt.hold(True)
    plt.scatter(data[100:,0], data[100:,1], c = 'blue')
    
    unlabeled_percentage = unlabeled_percentage_
    rng = np.random.RandomState(seed = seed_)
    unlabeled_idx = rng.rand(len(data)) < unlabeled_percentage
    
    partial_labels = labels.copy()
    partial_labels[unlabeled_idx] = -1
    
    label_prop_model = LabelPropagation(kernel = "rbf", gamma=20, 
                                        n_neighbors=7, max_iter=1000, tol=0.001)
    label_prop_model.fit(data, partial_labels)
    
    predicts = label_prop_model.predict(data[unlabeled_idx])
    total = predicts.shape[0]
    accu = np.where(labels[unlabeled_idx]==predicts)[0].shape[0]
    print("done, unlabel rate:", unlabeled_percentage_, "total:", total, "accu:", accu, "accuracy:", accu/total)
    
    

if __name__ == "__main__":
    for i in range(1, 10):
        demo_label_propgation_2002paper_same_distribution(i/10, 0)
        
    print("###################################################################")
    
    for i in range(1, 10):
        demo_label_propgation_2002paper_iris_dataset(i/10)