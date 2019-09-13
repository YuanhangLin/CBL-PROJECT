#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:43:23 2019

@author: linyuanhang
"""
import random
import numpy as np
from cvxopt import matrix, solvers
import math
import sklearn.manifold
from label_propagation_source_code_2008_paper import LocallyLinearEmbedding
import sklearn.neighbors
import matplotlib.pyplot as plt


def label_propagation_through_linear_neighborhoods(unlabeled_percentage_ = 0.1, seed_ = 0):
    
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
    unlabeled_idx = np.where((rng.rand(len(data)) < unlabeled_percentage)==True)[0]
    unlabeled_idx.astype(int)
    
    partial_labels = labels.copy()
    partial_labels[unlabeled_idx] = -1
    
    # build graph
#    graph_builder = sklearn.neighbors.NearestNeighbors(n_neighbors = 20, p = 2)
#    graph_builder.fit(data)
#    W = graph_builder.kneighbors_graph(data, n_neighbors = 20, mode = "distance").todense()
#    W = W / np.sum(W,axis = 1)
    
    my_LLE = LocallyLinearEmbedding(n_neighbors = 20, n_components = 1)
    my_LLE.fit(data)
    W = my_LLE.get_LLE_weight_matrix(data)
    
    
    print("...")
    
if __name__ == "__main__":
    label_propagation_through_linear_neighborhoods(unlabeled_percentage_ = 0.1, seed_ = 0)