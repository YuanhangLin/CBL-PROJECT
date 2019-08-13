#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from itertools import combinations
from hausdorff import hausdorff
from geopy.distance import geodesic
from math import sqrt
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy
import warnings
 

def graph_fourier_transform(A, normalized = "symmetric"):
    """
    This function applies Graph Convolution on a graph represented by adjacency matrix A
    by eigen-decomposition. This function takes adjacency matrix as input, and returns
    the eigenvectors and eigenvalues of the Laplacian matrix, and a Python dictionary 
    contains number of connected components and the conditional number of the Laplacian matrix.  
    
    Inputs:
    A         : V x V Numpy array, adjacency matrix
                V: number of vertices in the graph, 
    normalized: Python built-in string, normalization parameter
                "unnormalized" means without normalization
                "symmetric" means symmetric normalized
                "random" means random walk walk normalized
    
    Outputs:
    U         : V x V Numpy array, eigenvectors of the Laplacian matrix
    S         : V x 1 Numpy array, eigenvalues of the Laplacian matrix
    info      : Python built-in dictionary which contains number of connected components 
                and the condition number of the Laplacian matrix

    Reference:
    Eigendecomposition     :   https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix
    Laplacian matrix       :   https://en.wikipedia.org/wiki/Laplacian_matrix
    Graph fourier transform:   https://users.ece.cmu.edu/~asandryh/papers/icassp13a.pdf
    Positive semi-definite :   https://en.wikipedia.org/wiki/Definiteness_of_a_matrix
    """

    # sum the adjacency matrix by row to get the degree matrix 
    D = np.diag(np.sum(A, axis = 1)) 
    if normalized == "unnormalized":
        L = D - A
    elif normalized == "symmetric":
        D_inv_sqrt = np.sqrt(np.linalg.inv(D))
        I = np.eye(A.shape[0])
        L = I - np.linalg.multi_dot([D_inv_sqrt, A, D_inv_sqrt])
    elif normalized == "random":
        I = np.eye(A.shape[0])
        L = I - np.dot(np.linalg.inv(D), A)
    else:
        raise ValueError("Sorry. Unrecognized Normalization Type")
    
    if not np.all(np.linalg.eigvals(L) >= 0):
        # if L is not positive semi-definite
        # we apply diagonal loading to make it PSD
        warnings.warn("Laplacian matrix is not PSD, we apply Diagonal Loading to make it PSD")
        print("Before Diagonal Loading, the condition number of L is:", np.linalg.cond(L))
        L += 0.001*np.trace(L)*np.eye(L.shape[0])
        print("After Diagonal Loading, the condition number of L is:", np.linalg.cond(L))

    S, U = np.linalg.eig(L)

    # numpy sorts eigenvalues ascending by default
    # we want the eigenvalues sorted in descending order
    
    U = np.fliplr(U)
    S.sort()
    S = S[::-1]
    
    num_components = A.shape[0] - np.linalg.matrix_rank(L)
    condition_number = np.linalg.cond(L)
    info = {"num_components": num_components, 
            "condition_number":condition_number}
    
    return U, S, info

def polygon_aggregation(data, aggregation_method = None):
    """
    Call once, aggregates ONE polygon!

    This function aggregates multi-source data of pixels insides a polygon. 
    All pixels inside a polygon share same label, polygon_name, but have different data obtained by multiple sensors. 
    This function aggregates all the pixels that belong to the same polygon using aggregation_method.
    By aggregation, various-size polygons can be represented using several fixed-size feature vectors.
    
    Inputs:
    data               : (1) Single-sensor data: Npixel x D Numpy array, Npixel: number of pixels of a polygon 
                             D: number of features obtained by sensor
                         (2) Multi-sensor data: Python built-in list of N (Npixel x Di) Numpy arrays
                             N: number of sensors, Npixel: number of pixels of a polygon
                             Di: number of features obtained by sensor i, i = 1, 2, ... N
                         
    aggregation_method : Python function, the aggregation method which takes sensor data, i.e., 
                         N x 1 Numpy array of (Npixel x Di) Numpy arrays as input and returns aggregated data of the polygon
                         N x 1 Numpy array of (1 x Di) Numpy arrays
    
    Outputs:
    polygon_data       : (1) Single-sensor data: 1 x D Numpy array, D: number of features obtained by sensor
                         (2) Multi-sensor data: Python list contains N (1 x Di) Numpy arrays, aggregated data of each polygon
                             N: number of sensors, Di: number of features obtained by sensor i, i = 1, 2, ... N

    Reference:
    Different kinds of mean:   https://en.wikipedia.org/wiki/Mean
    """
    
    if type(data) is list:
        num_sensors = len(data)
    elif type(data) is np.ndarray:
        num_sensors = 1
    else:
        raise ValueError("Sorry. Unrecognized data type")
    polygon_data = [] if num_sensors > 1 else None
    for i in range(num_sensors):
        sensor_data = data[i] if num_sensors > 1 else data[:]
        sensor_data = aggregation_method(sensor_data)
        if num_sensors > 1:
            polygon_data.append(sensor_data)
        else: 
            polygon_data = sensor_data
    return polygon_data

def aggregate_using_unweighted_mean(sensor_data):
    """
    This function takes sensor data of pixels as input, aggregates by unweighted average,
    and returns aggregated sensor data of the polygon.
    
    Inputs:
    sensor_data:  Npixel x D Numpy arrays, Npixel: number of pixels of a polygon
                  D: number of features 

    Outputs:
    sensor_data:  1 x D Numpy arrays, D: number of features 

    """
    sensor_data = np.mean(sensor_data, axis = 0)
    sensor_data = sensor_data.reshape((1, sensor_data.shape[0]))
    return sensor_data

def aggregate_using_weighted_mean(sensor_data, ord = None):
    """
    This function takes sensor data of pixels as input, aggregates by weighted average(the weights 
    are the L2 norm of sensor data of each pixel), and returns aggregated sensor data of the polygon.
    
    Inputs:
    sensor_data:  Npixel x D Numpy arrays, Npixel: number of pixels of a polygon
                  D: number of features 
    ord        :  norm of matrices 
                  "None" means L2-norm
                  For more infomation please refer to numpy docs
                  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
                  
    Outputs:
    sensor_data:  1 x D Numpy arrays, D: number of features 

    Reference:
    Weighted arithmetic mean : https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    """

    weights = np.linalg.norm(sensor_data, ord = ord, axis = 1)
    weights = weights/sum(weights)
    weights = weights.reshape((sensor_data.shape[0],1))
    sensor_data = weights.T@sensor_data
    return sensor_data

def aggregate_using_truncated_mean(sensor_data, ord = None, alpha = 0.1):
    """
    This function takes sensor data of pixels as input, aggregates by weighted average(the weights 
    are the L2 norm of sensor data of each pixel), and returns aggregated sensor data of the polygon.
    
    Inputs:
    sensor_data:  Npixel x D Numpy arrays, Npixel: number of pixels of a polygon
                  D: number of features 
    ord        :  norm of matrices 
                  "None" means L2-norm
                  For more infomation please refer to numpy docs
                  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
    alpha      :  float, alpha-trimmed percentage used for truncated mean, in the range of [0, 0.5)

    Outputs:
    sensor_data:  1 x D Numpy arrays, Npixel: number of pixels of a polygon
                  D: number of features 

    Reference:
    Truncated mean : https://en.wikipedia.org/wiki/Truncated_mean

    """
    weights = np.linalg.norm(sensor_data)
    order = np.argsort(weights)
    order = order[int(len(order)*alpha):-int(len(order)*alpha)]
    sensor_data = np.mean(sensor_data[order,:], axis = 0)
    sensor_data = sensor_data.reshape(1, sensor_data.shape[0])
    return sensor_data

def construct_adjacency_matrix(dist, method = "fixed", n_neighbors = 10, 
                 sigma = 1.0, normalized = True, sparse = False):
    """
    This function takes distance matrix(dissimilarity matrix) as inputs and 
    returns adjacency matrix based on the distance matrix using KNN
    (fixed-size number KNN / radius-based KNN).
    
    Inputs:
    dist        : V x V Numpy array, distance matrix(dissimilarity matrix)
    method      : Python string, KNN implementation method
                 "radius" means constructing A by using radius-based KNN
                 "fixed"  means constructing A by using fixed-number KNN
    radius      : float value, radius used for RBF kernel
    n_neighbors : int value, number of nearest neighbors for fixed-number KNN
    sigma       : float value, magic number used for RBF kernel
    sparse      : bool, true means convert adjacency matrix to sparse matrix
    
    Outputs:
    adj          : V x V (Sparse) Numpy array, adjacency matrix

    Reference:
    K nearest neighbors algorithm: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
    """
    adj = np.zeros(dist.shape)
    if method == "radius":
        adj = np.exp(-(dist - dist.T)**2 / sigma**2)
        adj[np.isnan(adj)] = 0
    elif method == "fixed":
        for row in range(dist.shape[0]):
            temp = np.argsort(dist[row,:])[:n_neighbors] #indices
            if normalized:
                weights = dist[row, temp] / dist[row,temp[-1]] # in range[0,1]
                weights = np.exp(-weights/0.5)
                for i in range(len(temp)): 
                    adj[row, temp[i]] = weights[i]
    else:
        # currently we only support 2 ways of KNN
        raise ValueError("Sorry. Unrecognized KNN construction method")
    if sparse:
        adj = scipy.sparse.csr_matrix(adj)
    return adj
       
def construct_distance_matrix(data, polygon_names, metric = "Haudsdroff", norm = "L1", 
                 sensor = "AVIRIS"):
    """
    This function regards polygons as bags of several pixels and calculates pairwise Haudsdroff distances
    between polygons based on different metrics(e.g, L1/L2).
    
    Inputs:
    data          : Nsamples x D Numpy array, D: number of features
    polygon_names : Nsamples x 1 Numpy array
    Metric        : Python string, metric used for measuring pairwise distances
                    "Haudsdroff" means using Haudsdroff distance
    norm          : Python string
                    "L1": manhatthan distance
                    "L2": Euclidean distance
    sensor        : Python string, sensor type
                    "AVIRIS": hyperspectral data
                    "MASTER": thermal infrared data
                    "GIS"   : GIS data

    Outputs:
    dist          : Nsamples x Nsamples Numpy array, distance matrix

    Reference:
    L1, L2 norm           :  https://en.wikipedia.org/wiki/Norm_(mathematics)
    Haudsdroff distance   :  https://en.wikipedia.org/wiki/Hausdorff_distance
    """
    idx_mapping = {}
    polygons = np.array([]).astype(object)
    
    for polygon_name in polygon_names: 
        if polygon_name not in idx_mapping:
            idx_mapping[polygon_name] = np.where(polygon_name == polygon_names)[0]
            polygons = np.append(polygons, np.array(polygon_name))
            
    # for missing data, we regard the distance as inf, thus, they are not connected in the adjacency matrix
    dist = np.zeros((len(polygons), len(polygons)))*float('inf')
    
    if sensor == "AVIRIS" or sensor == "MASTER":
        if metric == "Haudsdroff":
            for pair in combinations(range(len(polygons)), 2):
                dist[pair[0], pair[1]] = hausdorff(data[idx_mapping[polygons[pair[0]]],:], 
                                                   data[idx_mapping[polygons[pair[1]]],:], distance = norm)
                dist[pair[1], pair[0]] = dist[pair[0], pair[1]]
        else:
            # currently we only use Haudsdroff distance --- multiple instance learning
            raise ValueError("Sorry. Unrecognized metric type")
    elif sensor == "GIS":
        # the first three features obtained by GIS are longtitde, latitude and elevation 
        long_lat, elevation = data[:,0:2], data[:, 2]
        for pair in combinations(range(len(polygons)), 2):
            # convert geodesic distance to euclidean distance
            long_lat_distance = geodesic(long_lat[idx_mapping[polygons[pair[0]]]], long_lat[idx_mapping[polygons[pair[1]]]]).km
            elevation_distance = abs(elevation[pair[0]] - elevation[pair[1]]) / 1000 # in km
            dist[pair[0], pair[1]] = sqrt(long_lat_distance**2 + elevation_distance**2) 
            dist[pair[1], pair[0]] = dist[pair[0], pair[1]]
    else:
        raise ValueError("Sorry. Unrecognized data sensor")
    
    return dist