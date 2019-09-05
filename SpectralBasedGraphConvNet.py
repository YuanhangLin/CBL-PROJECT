#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utilities import *
import numpy as np

class SpectralBasedGraphConvLayer(Module):
    """
    GraphConvLayer contains three variables:
    (1) _num_filters: A Python Integer, number of filters, which is equal to the number of input features
    (2) _filters    : A Pytorch Parameter contains a (_num_filters x 1) Pytorch FloatTensor
    (3) _bias       : A Pytorch Parameter contains a (_num_filters x 1) Pytorch FloatTensor, which is optional
    """
    
    def __init__(self, num_vertices, num_features, bias = False):
        super(SpectralBasedGraphConvLayer, self).__init__()
        self._num_filters = num_features
        self._num_vertices = num_vertices
        
        # According to Graph Fourier Transform, the filter is a V x V diagnoal matrix
        # Since the sensor has D features, the graph has D filter
        # 3D filters stored in 2D matrix, (D, V), each row is a set of filters for each feature
        # when do forward-propagation, convert each row(1xV vector) to a VxV diagnoal
        # thus, this (D x V) matrix is transformed to be D (VxV) matrices
        self._filters = Parameter(torch.FloatTensor(self._num_filters, self._num_vertices))
        if bias:
            self._bias = Parameter(torch.FloatTensor(self._num_filters, self._num_vertices))
        else:
            self.register_parameter("_bias", None)
        self.reset_parameters()
        
    def forward(self, x, U):
        """
        Inputs: 
        x            :   Nsamples x Nfeatures Pytorch FloatTensor
        U            :   Nsamples  x Nsamples Pytorch FloatTensor
        self._filters:   Nfeatures x Nsamples Pytorch FloatTensor
        
        Outputs:
        output       :   Nsamples x Nfeatures Pytorch FloatTensor
        """
        # z = x * w = UW(U^t)x
        # (U^t)U = I
        output = torch.zeros(x.shape).double()
        num_samples, num_features = x.shape
        x = x.double()
        U = U.double()
        for i in range(self._num_filters):
            W_i = torch.diag(self._filters[i, :]).double()
            y = torch.mm(U.t(), x[i,:])
            f = torch.mm(U, W_i)
            output[:,i] = torch.mm(f, y).t()
            if self._bias is not None:
                output[:,i] += self._bias[i,:].t()
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self._num_filters) + ') filters'
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self._filters.size(1))
        # the _filters is a diagonal matrix
        for i in range(len(self._filters)):
            self._filters[i,:,:].data.uniform_(-stdv, stdv)
            self._filters[i,:,:].data = torch.diag(torch.diag(self._filters[i,:,:].data)).double()
            if self._bias is not None:
                self._bias[i,:,:].data.uniform_(-stdv, stdv).double()
            
class SpectralBasedGraphConvNet(Module):
    
    def __init__(self, nfeat = [], nhid = [], adjs = [], nclass = 27):
        super(SpectralBasedGraphConvNet, self).__init__()
        
        # Graph Convolution Layers for 3 sensors
        self.num_vertices = adjs[0].shape[0]
        self._gc_spectra = SpectralBasedGraphConvLayer(self.num_vertices, nfeat[0])
        self._gc_thermal = SpectralBasedGraphConvLayer(self.num_vertices, nfeat[1])
        self._gc_gis = SpectralBasedGraphConvLayer(self.num_vertices, nfeat[2])
        
        # MLP for data-fusion and classification
        self._spectra_fc = nn.Linear(nfeat[0], nhid[0])
        self._thermal_fc = nn.Linear(nfeat[1], nhid[1])
        self._gis_fc = nn.Linear(nfeat[2], nhid[2])
        self._classifier = nn.Linear(sum(nhid), nclass)
        self._tanh = nn.Tanh()

        # graph_fourier_transform, import from utilities.py
        self.graph_fourier_transform = graph_fourier_transform
        adj_spectra, adj_thermal, adj_gis = adjs[0], adjs[1], adjs[2]
        self._U_spectra, _, _ = self.graph_fourier_transform(adj_spectra)
        self._U_thermal, _, _ = self.graph_fourier_transform(adj_thermal)
        self._U_gis, _, _ = self.graph_fourier_transform(adj_gis)
        
        # keep top num_filter eigenvectors
        self._U_spectra = torch.from_numpy(self._U_spectra.copy())
        self._U_thermal = torch.from_numpy(self._U_thermal.copy())
        self._U_gis = torch.from_numpy(self._U_gis.copy())
        
    def forward(self, data):
        spectra, thermal, gis = data[0], data[1], data[2]
        
        # Graph Convolution
        spectra = self._gc_spectra.forward(spectra, self._U_spectra).t()
        thermal = self._gc_thermal.forward(thermal, self._U_thermal).t()
        gis = self._gc_gis.forward(gis, self._U_gis).t()
        
        # MLP learning
        spectra = self._spectra_fc.forward(spectra.float())
        thermal = self._thermal_fc.forward(thermal.float())
        gis = self._gis_fc.forward(gis.float())
        
        # feature-level data fusion for joint understanding 
        output = torch.cat((spectra, thermal, gis), 1)
        output = self._classifier(output.float())
        
        return output
    
    def save_state_to_file(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load_state_from_file(self, filepath):
        self.load_state_dict(torch.load(filepath))
    
if __name__ == "__main__":
    net = SpectralBasedGraphConvNet([26, 4, 7], [20, 2, 4], [np.ones((1000,1000))]*3,nclass=27)
    data = [torch.rand(1000,26), torch.rand(1000,4), torch.rand(1000,7)]