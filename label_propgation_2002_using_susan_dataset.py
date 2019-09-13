#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:15:19 2019

@author: linyuanhang
"""
import numpy as np
from scipy.io import loadmat
from sklearn.semi_supervised import LabelPropagation


def demo_pixel_level(seed = 0):
    x = loadmat("../data/SusanSpectraProcessed130411_classesremoved.mat", squeeze_me = True)
    data, labels = x["spectra"], x["labels"]
    label_prop_model = LabelPropagation(kernel = "rbf", gamma=20, 
                                        n_neighbors=20, max_iter=1000, tol=0.001)
    rng = np.random.RandomState(seed)
    
    for i in range(5, 100, 5):
        print("start working", i)
        unlabeled_percentage = i/100
        unlabeled_idx = np.where((rng.rand(len(data)) < unlabeled_percentage) == True)[0]
        
        partial_labels = labels.copy()
        partial_labels[unlabeled_idx] = -1
        
        label_prop_model = LabelPropagation(kernel = "rbf", gamma=20, 
                                            n_neighbors=7, max_iter=1000, tol=0.001)
        
        label_prop_model.fit(data, partial_labels)
        predicts = label_prop_model.predict(data[unlabeled_idx])
        total = predicts.shape[0]
        accu = np.where(labels[unlabeled_idx]==predicts)[0].shape[0]
        
        print("done, unlabel rate:", unlabeled_percentage, "accuracy:", accu/total)
        
def demo_polygon_level_truncated_aggregation_balanced_unlabeled(seed = 0):
    x = loadmat("../data/exp1_130411_aggregated_dataset.mat", squeeze_me = True)
    data, labels = x["polygon_spectra"], x["polygon_labels"]
    label_prop_model = LabelPropagation(kernel = "rbf", gamma=20, 
                                        n_neighbors=20, max_iter=1000, tol=0.001)
    rng = np.random.RandomState(seed)
    
    for i in range(5, 100, 5):
        print("start working", i)
        unlabeled_percentage = i/100
        partial_labels = labels.copy()
        
        for c in range(27):
            c_idx = np.where(labels == c)[0]
            unlabeled_idx = np.where((rng.rand(len(c_idx)) < unlabeled_percentage) == True)[0]
            partial_labels[c_idx[unlabeled_idx]] = -1
        
        label_prop_model = LabelPropagation(kernel = "rbf", gamma=20, 
                                            n_neighbors=7, max_iter=1000, tol=0.001)
        
        label_prop_model.fit(data, partial_labels)
        predicts = label_prop_model.predict(data[unlabeled_idx])
        total = predicts.shape[0]
        accu = np.where(labels[unlabeled_idx]==predicts)[0].shape[0]
        
        print("balanced unlabeled percentage:", unlabeled_percentage, "accuracy:", accu/total)

def demo_polygon_level_truncated_aggregation_unbalanced_unlabeled(seed = 0):
    x = loadmat("../data/exp1_130411_aggregated_dataset.mat", squeeze_me = True)
    data, labels = x["polygon_spectra"], x["polygon_labels"]
    label_prop_model = LabelPropagation(kernel = "rbf", gamma=20, 
                                        n_neighbors=20, max_iter=1000, tol=0.001)
    rng = np.random.RandomState(seed)
    
    for i in range(5, 100, 5):
        print("start working", i)
        unlabeled_percentage = i/100
        partial_labels = labels.copy()
        
        unlabeled_idx = np.where((rng.rand(len(partial_labels)) < unlabeled_percentage) == True)[0]
        partial_labels[unlabeled_idx] = -1
        
        label_prop_model = LabelPropagation(kernel = "rbf", gamma=20, 
                                            n_neighbors=7, max_iter=1000, tol=0.001)
        
        label_prop_model.fit(data, partial_labels)
        predicts = label_prop_model.predict(data[unlabeled_idx])
        total = predicts.shape[0]
        accu = np.where(labels[unlabeled_idx]==predicts)[0].shape[0]
        
        print("unbalanced unlabeled percentage:", unlabeled_percentage, "accuracy:", accu/total)


if __name__ == "__main__":
#    demo_pixel_level()
    demo_polygon_level_truncated_aggregation_balanced_unlabeled()
    print("#######################################################")
    demo_polygon_level_truncated_aggregation_unbalanced_unlabeled()