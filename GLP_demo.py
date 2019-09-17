#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:59:32 2019

@author: yuanhang
"""

import numpy as np
from GLP import GLP
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.io import loadmat


def demo_separate_gaussian(seed_ = 0):
    
    mu_1 = np.array([1, 1])
    sigma_1 = 0.01 * np.diag(np.ones(2))
    data_class_1 = np.random.multivariate_normal(mu_1, sigma_1, 100)
    labels_class_1 = np.zeros(100).astype(int)
    
    mu_2 = np.array([-1, -1])
    sigma_2 = 0.1 * np.diag(np.ones(2))
    data_class_2 = np.random.multivariate_normal(mu_2, sigma_2, 100)
    labels_class_2 = np.ones(100).astype(int)
    
    data = np.vstack((data_class_1, data_class_2))
    labels = np.concatenate((labels_class_1, labels_class_2))
    
    num_exps = int((95-5)/5 + 1)
    results = np.zeros((num_exps, 4)) # num_data, num_labeled_data, unlabeled_per, accu
    
    result_folder = "GLP_demo_separate_gaussian/"
    
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    
    for i in range(5, 100, 5):
    
        unlabeled_percentage = i/100
        rng = np.random.RandomState(seed = seed_)
        unlabeled_idx = np.where((rng.rand(len(data)) <= unlabeled_percentage)==True)[0]
        unlabeled_idx.astype(int)
        
        partial_labels = labels.copy()
        partial_labels[unlabeled_idx] = -1
        
        label_prop_model = GLP(kernel = "rbf", gamma=20, 
                                            n_neighbors=7, max_iter=1000, tol=0.001)
        label_prop_model.fit(data, partial_labels)
        
        predicts = label_prop_model.predict(data[unlabeled_idx])
        total = predicts.shape[0]
        accu = np.where(labels[unlabeled_idx]==predicts)[0].shape[0]
        
        plt.figure(figsize=(12,8))
        
        plt.scatter(data[0:100,0], data[0:100,1], c = 'red',  marker = 'x')
        plt.scatter(data[100:,0], data[100:,1], c = 'blue',  marker = 'x')
        unlabeld_class1_idx = unlabeled_idx[np.where(labels[unlabeled_idx] == 0)[0]]
        unlabeld_class2_idx = unlabeled_idx[np.where(labels[unlabeled_idx] == 1)[0]]
        
        plt.scatter(data[unlabeld_class1_idx, 0], data[unlabeld_class1_idx, 1], s = 80, facecolors='none', edgecolors='r')
        plt.scatter(data[unlabeld_class2_idx, 0], data[unlabeld_class2_idx, 1], s = 80, facecolors='none', edgecolors='b')
        
        plt.title("circle:unlabeled data, mark:labeled data, unlabeled rate:" + str(round(unlabeled_percentage * 100)) + "%, accu:" + str(round(accu/total * 100)) + "%")
        print("done, unlabel rate:", unlabeled_percentage, "total:", total, "accu:", accu, "accuracy:", accu/total)
        
        os.chdir(result_folder)
        fig_name = "unlabeled_percentage_" + str(i)
        plt.savefig(fig_name)
        os.chdir("../")
        
        plt.show()
        
        results[int((i-5)/5),0] = data.shape[0]
        results[int((i-5)/5),1] = np.where(partial_labels != -1)[0].shape[0]
        results[int((i-5)/5),2] = round(unlabeled_percentage * 100)
        results[int((i-5)/5),3] = round(accu/total * 100)
    
    os.chdir(result_folder)
    np.savetxt("GLP_separate_gaussian.csv", results, fmt = "%d", delimiter='\t', 
               header = "num_data\tnum_labeled_data\tunlabeled_percentage\taccuracy")
    os.chdir("../")

def demo_mixed_gaussian(seed_ = 0):
    
    mu_1 = np.array([0.8, 0.8])
    sigma_1 = 0.01 * np.diag(np.ones(2))
    data_class_1 = np.random.multivariate_normal(mu_1, sigma_1, 100)
    labels_class_1 = np.zeros(100).astype(int)
    
    mu_2 = np.array([0.5, 0.5])
    sigma_2 = 0.05 * np.diag(np.ones(2))
    data_class_2 = np.random.multivariate_normal(mu_2, sigma_2, 100)
    labels_class_2 = np.ones(100).astype(int)
    
    data = np.vstack((data_class_1, data_class_2))
    labels = np.concatenate((labels_class_1, labels_class_2))
    
    num_exps = int((95-5)/5 + 1)
    results = np.zeros((num_exps, 4)) # num_data, num_labeled_data, unlabeled_per, accu
    
    result_folder = "GLP_demo_mixed_gaussian/"
    
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    
    for i in range(5, 100, 5):
    
        unlabeled_percentage = i/100
        rng = np.random.RandomState(seed = seed_)
        unlabeled_idx = np.where((rng.rand(len(data)) <= unlabeled_percentage)==True)[0]
        unlabeled_idx.astype(int)
        
        partial_labels = labels.copy()
        partial_labels[unlabeled_idx] = -1
        
        label_prop_model = GLP(kernel = "rbf", gamma=20, 
                                            n_neighbors=20, max_iter=10000, tol=0.001)
        label_prop_model.fit(data, partial_labels)
        
        predicts = label_prop_model.predict(data[unlabeled_idx])
        total = predicts.shape[0]
        accu = np.where(labels[unlabeled_idx]==predicts)[0].shape[0]
        
        plt.figure(figsize=(12,8))
        
        plt.scatter(data[0:100,0], data[0:100,1], c = 'red',  marker = 'x')
        plt.scatter(data[100:,0], data[100:,1], c = 'blue',  marker = 'x')
        unlabeld_class1_idx = unlabeled_idx[np.where(labels[unlabeled_idx] == 0)[0]]
        unlabeld_class2_idx = unlabeled_idx[np.where(labels[unlabeled_idx] == 1)[0]]
        
        plt.scatter(data[unlabeld_class1_idx, 0], data[unlabeld_class1_idx, 1], s = 80, facecolors='none', edgecolors='r')
        plt.scatter(data[unlabeld_class2_idx, 0], data[unlabeld_class2_idx, 1], s = 80, facecolors='none', edgecolors='b')
        
        plt.title("circle:unlabeled data, mark:labeled data, unlabeled rate:" + str(round(unlabeled_percentage * 100)) + "%, accu:" + str(round(accu/total * 100)) + "%")
        print("done, unlabel rate:", unlabeled_percentage, "total:", total, "accu:", accu, "accuracy:", accu/total)
        
        os.chdir(result_folder)
        fig_name = "unlabeled_percentage_" + str(i)
        plt.savefig(fig_name)
        os.chdir("../")
        
        plt.show()
        
        results[int((i-5)/5),0] = data.shape[0]
        results[int((i-5)/5),1] = np.where(partial_labels != -1)[0].shape[0]
        results[int((i-5)/5),2] = round(unlabeled_percentage * 100)
        results[int((i-5)/5),3] = round(accu/total * 100)

    os.chdir(result_folder)
    np.savetxt("GLP_mixed_gaussian.csv", results, fmt = "%d", delimiter='\t', 
               header = "num_data\tnum_labeled_data\tunlabeled_percentage\taccuracy")
    os.chdir("../")


def demo_polygon_level_truncated_aggregation_balanced_unlabeled(seed = 0):
    x = loadmat("exp1_130411_aggregated_dataset.mat", squeeze_me = True)
    data, labels = x["polygon_spectra"], x["polygon_labels"]

    rng = np.random.RandomState(seed)
    
    size = int((95-5)/5 + 1)
    results = np.zeros((size, 4))
    
    result_folder = "GLP_demo_susan_dataset_balanced/"
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    
    for i in range(5, 100, 5):
        print("start working", i)
        unlabeled_percentage = i/100
        partial_labels = labels.copy()
        
        for c in range(27):
            c_idx = np.where(labels == c)[0]
            unlabeled_idx = np.where((rng.rand(len(c_idx)) < unlabeled_percentage) == True)[0]
            partial_labels[c_idx[unlabeled_idx]] = -1
        
        label_prop_model = GLP(kernel = "rbf", gamma=20, 
                                            n_neighbors=7, max_iter=1000, tol=0.001)
        
        label_prop_model.fit(data, partial_labels)
        predicts = label_prop_model.predict(data[unlabeled_idx])
        total = predicts.shape[0]
        accu = np.where(labels[unlabeled_idx]==predicts)[0].shape[0]
        
        print("polygon_level_truncated_aggregation_balanced_unlabeled percentage:", unlabeled_percentage, "accuracy:", accu/total)
        
        results[int((i-5)/5), 0] = data.shape[0]
        results[int((i-5)/5), 1] = np.where(partial_labels != -1)[0].shape[0]
        results[int((i-5)/5), 2] = round(unlabeled_percentage * 100)
        results[int((i-5)/5), 3] = round(accu/total*100)
    
    os.chdir(result_folder)
    np.savetxt("GLP_susan_dataset_balanced_unlabeled.csv", results, fmt = "%d", delimiter='\t', 
               header = "num_data\tnum_labeled_data\tunlabeled_percentage\taccuracy")
    os.chdir("../")

def demo_polygon_level_truncated_aggregation_unbalanced_unlabeled(seed = 0):
    x = loadmat("exp1_130411_aggregated_dataset.mat", squeeze_me = True)
    data, labels = x["polygon_spectra"], x["polygon_labels"]

    rng = np.random.RandomState(seed)
    
    result_folder = "GLP_demo_susan_dataset_unbalanced/"
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    
    size = int((95-5)/5 + 1)
    results = np.zeros((size, 4))
    
    for i in range(5, 100, 5):
        print("start working", i)
        unlabeled_percentage = i/100
        partial_labels = labels.copy()
        
        unlabeled_idx = np.where((rng.rand(len(partial_labels)) < unlabeled_percentage) == True)[0]
        partial_labels[unlabeled_idx] = -1
        
        label_prop_model = GLP(kernel = "rbf", gamma=20, 
                                            n_neighbors=7, max_iter=1000, tol=0.001)
        
        label_prop_model.fit(data, partial_labels)
        predicts = label_prop_model.predict(data[unlabeled_idx])
        total = predicts.shape[0]
        accu = np.where(labels[unlabeled_idx]==predicts)[0].shape[0]
        
        print("polygon_level_truncated_aggregation_unbalanced_unlabeled percentage:", unlabeled_percentage, "accuracy:", accu/total)

        results[int((i-5)/5), 0] = data.shape[0]
        results[int((i-5)/5), 1] = np.where(partial_labels != -1)[0].shape[0]
        results[int((i-5)/5), 2] = round(unlabeled_percentage * 100)
        results[int((i-5)/5), 3] = round(accu/total*100)
        

    os.chdir(result_folder)
    np.savetxt("GLP_susan_dataset_unbalanced_unlabeled.csv", results, fmt = "%d", delimiter='\t', 
               header = "num_data\tnum_labeled_data\tunlabeled_percentage\taccuracy")
    os.chdir("../")


if __name__ == "__main__":
    demo_separate_gaussian(0)
    demo_mixed_gaussian(0)
    demo_polygon_level_truncated_aggregation_balanced_unlabeled(seed = 0)
    demo_polygon_level_truncated_aggregation_unbalanced_unlabeled(seed = 0)