#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utilities import *
import numpy as np

class GraphConvLayer(Module):
    """
    GraphConvLayer contains three variables:
    (1) _num_filters: A Python Integer, number of filters, which is equal to the number of input features
    (2) _filters    : A Pytorch Parameter contains a (_num_filters x 1) Pytorch FloatTensor
    (3) _bias       : A Pytorch Parameter contains a (_num_filters x 1) Pytorch FloatTensor, which is optional
    """
    
    def __init__(self, num_features, bias = False):
        super(GraphConvLayer, self).__init__()
        self._num_filters = num_features
        self._filters = Parameter(torch.FloatTensor(self._num_filters, self._num_filters))
        if bias:
            self._bias = Parameter(torch.FloatTensor(self._num_filters))
        else:
            self.register_parameter("_bias", None)
        self.reset_parameters()
        
    def forward(self, x, U):
        x = torch.mm(U.t(), x)
        f = torch.mm(U, self._filters.data)
        output = torch.mm(f, x)
        return output + self._bias if (self._bias is not None) else output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self._num_filters) + ') filters'
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self._filters.size(1))
        self._filters.data.uniform_(-stdv, stdv)
        # the _filters is a diagonal matrix
        self._filters.data = torch.diag(torch.diag(self._filters.data))
        if self._bias is not None:
            self._bias.data.uniform_(-stdv, stdv)
            
class GraphConvNet(Module):
    
    def __init__(self, nfeat = [], nhid = [], adjs = [], nclass = 27):
        super(GraphConvNet, self).__init__()
        
        # Graph Convolution Layers for 3 sensors
        self._gc_spectra = GraphConvLayer(nfeat[0])
        self._gc_thermal = GraphConvLayer(nfeat[1])
        self._gc_gis = GraphConvLayer(nfeat[2])
        
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
        self._U_spectra = self._U_spectra[:, : nfeat[0]]
        self._U_thermal = self._U_thermal[:, : nfeat[1]]
        self._U_gis = self._U_gis[:, : nfeat[2]]
        
    def forward(self, data):
        spectra, thermal, gis = data[0], data[1], data[2]
        
        # Graph Convolution
        spectra = self._gc_spectra.forward(spectra, self._U_spectra)
        thermal = self._gc_thermal.forward(thermal, self._U_thermal)
        gis = self._gc_gis.forward(gis, self._U_gis)
        
        # MLP learning
        spectra = self._spectra_fc.forward(spectra)
        thermal = self._thermal_fc.forward(thermal)
        gis = self._gis_fc.forward(gis)
        
        # feature-level data fusion for joint understanding 
        output = torch.cat((spectra, thermal, gis), 1)
        output = self._classifier(output)
        
        return output
    
    def save_state_to_file(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load_state_from_file(self, filepath):
        self.load_state_dict(torch.load(filepath))