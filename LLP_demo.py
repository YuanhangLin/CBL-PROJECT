#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:58:57 2019

@author: linyuanhang
"""

import random
import numpy as np
from cvxopt import matrix, solvers
import os
import math
import matplotlib.pyplot as plt
from LLP import *
from sklearn import datasets

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
    
    total = len(data)
    
    num_exps = int((95-5)/5 + 1)
    results = np.zeros((num_exps, 4)) # num_data, num_labeled_data, unlabeled_per, accu
    
    result_folder = "LLP_demo_separate_gaussian/"
    
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    
    rng = np.random.RandomState(seed = seed_)
    
    for i in range(5, 100, 5):
    
        unlabeled_percentage = i/100
        unlabeled_idx = np.where((rng.rand(len(data)) <= unlabeled_percentage)==True)[0]
        unlabeled_idx.astype(int)
        
        partial_labels = labels.copy()
        partial_labels[unlabeled_idx] = -1
        
        label_prop_model = LLP(data = data, partial_labels = partial_labels, true_labels = labels, 
                               alpha = 0.99, max_iter = 100, tol = 0.001, n_neighbors = 5)
        
        label_prop_model.LNPiter()
        
        accu = label_prop_model.calc_accuracy()
        
        plt.figure(figsize=(12,8))
        
        plt.scatter(data[0:100,0], data[0:100,1], c = 'red',  marker = 'x')
        plt.scatter(data[100:,0], data[100:,1], c = 'blue',  marker = 'x')
        unlabeld_class1_idx = unlabeled_idx[np.where(labels[unlabeled_idx] == 0)[0]]
        unlabeld_class2_idx = unlabeled_idx[np.where(labels[unlabeled_idx] == 1)[0]]
        
        plt.scatter(data[unlabeld_class1_idx, 0], data[unlabeld_class1_idx, 1], s = 80, facecolors='none', edgecolors='r')
        plt.scatter(data[unlabeld_class2_idx, 0], data[unlabeld_class2_idx, 1], s = 80, facecolors='none', edgecolors='b')
        
        plt.title("circle:unlabeled data, mark:labeled data, unlabeled rate:" + str(round(unlabeled_percentage * 100)) + "%, accu:" + str(round(accu * 100)) + "%")
        print("done, unlabel rate:", unlabeled_percentage, "total:", total, "accu:", accu)
        
        os.chdir(result_folder)
        fig_name = "unlabeled_percentage_" + str(i)
        plt.savefig(fig_name)
        os.chdir("../")
        
        plt.show()
        
        results[int((i-5)/5),0] = data.shape[0]
        results[int((i-5)/5),1] = np.where(partial_labels != -1)[0].shape[0]
        results[int((i-5)/5),2] = round(unlabeled_percentage * 100)
        results[int((i-5)/5),3] = round(accu * 100)
    
    os.chdir(result_folder)
    np.savetxt("LLP_separate_gaussian.csv", results, fmt = "%d", delimiter='\t', 
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
    
    total = len(data)
    
    num_exps = int((95-5)/5 + 1)
    results = np.zeros((num_exps, 4)) # num_data, num_labeled_data, unlabeled_per, accu
    
    result_folder = "LLP_demo_mixed_gaussian/"
    
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    
    rng = np.random.RandomState(seed = seed_)
    
    for i in range(5, 100, 5):
    
        unlabeled_percentage = i/100
        unlabeled_idx = np.where((rng.rand(len(data)) <= unlabeled_percentage)==True)[0]
        unlabeled_idx.astype(int)
        
        partial_labels = labels.copy()
        partial_labels[unlabeled_idx] = -1
        
        label_prop_model = LLP(data = data, partial_labels = partial_labels, true_labels = labels, 
                               alpha = 0.99, max_iter = 1000, tol = 0.00001)
        
        label_prop_model.LNPiter()
        
        accu = label_prop_model.calc_accuracy()
        
        plt.figure(figsize=(12,8))
        
        plt.scatter(data[0:100,0], data[0:100,1], c = 'red',  marker = 'x')
        plt.scatter(data[100:,0], data[100:,1], c = 'blue',  marker = 'x')
        unlabeld_class1_idx = unlabeled_idx[np.where(labels[unlabeled_idx] == 0)[0]]
        unlabeld_class2_idx = unlabeled_idx[np.where(labels[unlabeled_idx] == 1)[0]]
        
        plt.scatter(data[unlabeld_class1_idx, 0], data[unlabeld_class1_idx, 1], s = 80, facecolors='none', edgecolors='r')
        plt.scatter(data[unlabeld_class2_idx, 0], data[unlabeld_class2_idx, 1], s = 80, facecolors='none', edgecolors='b')
        
        plt.title("circle:unlabeled data, mark:labeled data, unlabeled rate:" + str(round(unlabeled_percentage * 100)) + "%, accu:" + str(round(accu * 100)) + "%")
        print("done, unlabel rate:", unlabeled_percentage, "total:", total, "accu:", accu)
        
        os.chdir(result_folder)
        fig_name = "unlabeled_percentage_" + str(i)
        plt.savefig(fig_name)
        os.chdir("../")
        
        plt.show()
        
        results[int((i-5)/5),0] = data.shape[0]
        results[int((i-5)/5),1] = np.where(partial_labels != -1)[0].shape[0]
        results[int((i-5)/5),2] = round(unlabeled_percentage * 100)
        results[int((i-5)/5),3] = round(accu * 100)
    
    os.chdir(result_folder)
    np.savetxt("LLP_mixed_gaussian.csv", results, fmt = "%d", delimiter='\t', 
               header = "num_data\tnum_labeled_data\tunlabeled_percentage\taccuracy")
    os.chdir("../")

    
if __name__ == "__main__":
    demo_separate_gaussian(seed_ = 0)
    demo_mixed_gaussian(seed_ = 0)