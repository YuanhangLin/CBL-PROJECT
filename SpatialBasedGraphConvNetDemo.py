#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from utilities import *
from SpatialBasedGraphConvNet import SpatialBasedGraphConvNet
from scipy.io import loadmat
import numpy as np
import torch.utils.data as tdata

class Dataset(tdata.Dataset):
    
    def __init__(self, root = "../data/", date = "130411"):
        """
        state : 0 means testing, 1 means training, 2 means valid
        """
        file_name = root + "exp1_" + date + "_aggregated_dataset.mat"
        x = loadmat(file_name, squeeze_me = True)
        self._polygon_names = x["unique_polygons"][:, np.newaxis].astype(str)
        self._spectra = x["polygon_spectra"]
        self._thermal = x["polygon_thermal"]
        self._gis = x["polygon_gis"]
        self._labels = x["polygon_labels"][:, np.newaxis].astype(np.int)
        self._spectra_missing_flag = x["spectra_missing_flag"]
        self._thermal_missing_flag = x["thermal_missing_flag"]
        self._gis_missing_flag = x["gis_missing_flag"]
        self._missing_flag = x["missing_data_flag"].T
        
    def __len__(self):
        return len(self._polygon_names)
    
    def __getitem__(self, index):
        return torch.from_numpy(self._spectra[index,:]).float(), \
               torch.from_numpy(self._thermal[index,:]).float() , \
               torch.from_numpy(self._gis[index,:]).float(), \
               torch.from_numpy(self._labels[index])
               
def demo():
    # parameter setting
    num_epochs = 3000
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss() # object function
    
    # load data
    dataset = Dataset(date = "130411")
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size = len(dataset), shuffle = True)
    
    # build dists
    dist_spectra = construct_distance_matrix_using_aggregated_polygon(dataset._spectra)
    dist_thermal = construct_distance_matrix_using_aggregated_polygon(dataset._thermal)
    dist_gis = construct_distance_matrix_using_aggregated_polygon(dataset._gis)
    
    # build adjs
    adj_spectra = construct_adjacency_matrix(dist_spectra, method = "fixed", n_neighbors = 30)
    adj_thermal = construct_adjacency_matrix(dist_thermal, method = "fixed", n_neighbors = 10)
    adj_gis = construct_adjacency_matrix(dist_gis, method = "fixed", n_neighbors = 20)
    
    # initilize network & optimizer
    nfeat = [dataset._spectra.shape[1], dataset._thermal.shape[1], dataset._gis.shape[1]]
    nhid = [44, 2, 4]
    adjs = [adj_spectra, adj_thermal, adj_gis]
    nclass = 27
    
    net = SpatialBasedGraphConvNet(nfeat = nfeat, nhid = nhid, nclass = nclass)
    net.cuda() # use GPU to acclerate training/testing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    
    average_loss, ctr = 0, 0
    
    for i in range(len(adjs)):
        adjs[i] = torch.Tensor(adjs).float().to(device)
        adjs[i] = torch.autograd.Variable(adjs[i])
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for _, data in enumerate(dataLoader, 0):
            spectra, thermal, gis, labels = data[0].to(device), data[1].to(device), \
                                                          data[2].to(device), data[3].to(device)
            spectra, thermal, gis, labels = torch.autograd.Variable(spectra), torch.autograd.Variable(thermal), \
                                                          torch.autograd.Variable(gis), torch.autograd.Variable(labels)
            optimizer.zero_grad()
            x = [spectra, thermal, gis]
            output = net.forward(x, adjs)
            loss = criterion(output, labels.squeeze(0) if output.shape[0] == 1 else labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            _, predicted = torch.max(output, 1)
            running_loss = 0.0
    
    