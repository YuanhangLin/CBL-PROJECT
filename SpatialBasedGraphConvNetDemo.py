#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from utilities import construct_distance_matrix_using_aggregated_polygon, construct_adjacency_matrix
from SpatialBasedGraphConvNet import SpatialBasedGraphConvNet
from scipy.io import loadmat
import numpy as np
import torch.utils.data as tdata

class Dataset(tdata.Dataset):
    
    def __init__(self, root = "../data/", date = "130411", state = 0):
        """
        state : 0 means testing, 1 means training, 2 means valid
        """
        file_name = root + "exp1_" + date + "_aggregated_dataset.mat"
        x = loadmat(file_name, squeeze_me = True)
        self._indices = np.where(x["split_indices"] == state)[0]
        self._polygon_names = x["unique_polygons"][:, np.newaxis].astype(str)
        self._polygon_names = self._polygon_names[self._indices]
        self._spectra = x["polygon_spectra"][self._indices,:]
        self._thermal = x["polygon_thermal"][self._indices,:]
        self._gis = x["polygon_gis"][self._indices,:]
        self._labels = x["polygon_labels"][self._indices, np.newaxis].astype(np.int)
        
    def __len__(self):
        return len(self._polygon_names)
    
    def __getitem__(self, index):
        return torch.from_numpy(self._spectra[index,:]).float(), \
               torch.from_numpy(self._thermal[index,:]).float() , \
               torch.from_numpy(self._gis[index,:]).float(), \
               torch.from_numpy(self._labels[index])
               
def demo():
    # parameter setting
    num_epochs = 100000
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss() # object function
    
    # load data
    trainset = Dataset(date = "130411", state = 1)
    trainingLoader = torch.utils.data.DataLoader(trainset, batch_size = len(trainset), shuffle = True)
    
    # build dists
    dist_spectra = construct_distance_matrix_using_aggregated_polygon(trainset._spectra)
    dist_thermal = construct_distance_matrix_using_aggregated_polygon(trainset._thermal)
    dist_gis = construct_distance_matrix_using_aggregated_polygon(trainset._gis)
    
    # build adjs
    adj_spectra = construct_adjacency_matrix(dist_spectra, method = "fixed", n_neighbors = 30)
    adj_thermal = construct_adjacency_matrix(dist_thermal, method = "fixed", n_neighbors = 10)
    adj_gis = construct_adjacency_matrix(dist_gis, method = "fixed", n_neighbors = 20)
    
    # initilize network & optimizer
    nfeat = [trainset._spectra.shape[1], trainset._thermal.shape[1], trainset._gis.shape[1]]
    nhid = [44, 2, 4]
    adjs = [adj_spectra, adj_thermal, adj_gis]
    nclass = 27
    
    net = SpatialBasedGraphConvNet(nfeat = nfeat, nhid = nhid, nclass = nclass)
    
    net.cuda() # use GPU to acclerate training/testing
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net.train() # set the net to training mode 
    
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    
    average_loss, ctr = 0, 0
    
    for i in range(len(adjs)):
        # use gpu
        adjs[i] = torch.Tensor(adjs[i]).float().to(device)
        adjs[i] = torch.autograd.Variable(adjs[i])
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for _, data in enumerate(trainingLoader, 0):
            # set all variables to GPU
            spectra, thermal, gis, labels = data[0].to(device), data[1].to(device), \
                                                          data[2].to(device), data[3].to(device)
            spectra, thermal, gis, labels = torch.autograd.Variable(spectra), torch.autograd.Variable(thermal), \
                                                          torch.autograd.Variable(gis), torch.autograd.Variable(labels)
            
            # clear previous gradients
            optimizer.zero_grad()
            
            # create multi-source input
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
    
    
    net.eval()
    with torch.no_grad():
        testset = Dataset(date = "130411", state = 0)
        # build dists
        dist_spectra = construct_distance_matrix_using_aggregated_polygon(testset._spectra)
        dist_thermal = construct_distance_matrix_using_aggregated_polygon(testset._thermal)
        dist_gis = construct_distance_matrix_using_aggregated_polygon(testset._gis)
        
        # build adjs
        adj_spectra = construct_adjacency_matrix(dist_spectra, method = "fixed", n_neighbors = 30)
        adj_thermal = construct_adjacency_matrix(dist_thermal, method = "fixed", n_neighbors = 10)
        adj_gis = construct_adjacency_matrix(dist_gis, method = "fixed", n_neighbors = 20)
        adjs = [adj_spectra, adj_thermal, adj_gis]
        
        for i in range(len(adjs)):
            # use gpu
            adjs[i] = torch.Tensor(adjs[i]).float().to(device)
            adjs[i] = torch.autograd.Variable(adjs[i])
        
        testingLoader = torch.utils.data.DataLoader(testset, batch_size = len(testset), shuffle = True)
        for _, data in enumerate(testingLoader, 0):
            spectra, thermal, gis, labels = data[0].to(device), data[1].to(device), \
                                                          data[2].to(device), data[3].to(device)
            spectra, thermal, gis, labels = torch.autograd.Variable(spectra), torch.autograd.Variable(thermal), \
                                                          torch.autograd.Variable(gis), torch.autograd.Variable(labels)
            
            # clear previous gradients
            optimizer.zero_grad()
            x = [spectra, thermal, gis]
            output = net.forward(x, adjs)
            _, predicted = torch.max(output.data, 1)
            labels = labels.squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

if __name__ == "__main__":
    demo()