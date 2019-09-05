#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utilities import *
import numpy as np
import torch.nn.functional as F

class SpatialBasedGraphConvLayer(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(SpatialBasedGraphConvLayer, self).__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self._bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("_bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self._weight.size(1))
        self._weight.data.uniform_(-stdv, stdv)
        if self._bias is not None:
            self._bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self._weight)
        output = torch.spmm(adj, support)
        if self._bias is not None:
            return output + self._bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self._in_features) + ' -> ' \
               + str(self._out_features) + ')'
        
class SpatialBasedGraphConvNet(Module):
    
    def __init__(self, nfeat = [], nhid = [], nclass = 27, dropout = 0.1):
        
        super(SpatialBasedGraphConvNet, self).__init__()

        # Graph Convolution Layers for 3 sensors
        self._gc_spectra = SpatialBasedGraphConvLayer(nfeat[0], nhid[0])
        self._gc_thermal = SpatialBasedGraphConvLayer(nfeat[1], nhid[1])
        self._gc_gis = SpatialBasedGraphConvLayer(nfeat[2], nhid[2])
        
        # MLP for data-fusion and classification
        self._classifier = nn.Linear(sum(nhid), nclass)

        self.dropout = dropout

    def forward(self, x, adjs):
        spectra, thermal, gis = x[0], x[1], x[2]
        adj_spectra, adj_thermal, adj_gis = adjs[0], adjs[1], adjs[2]
        
        # remove all nan
        adj_spectra[torch.isnan(adj_spectra)] = 0
        adj_thermal[torch.isnan(adj_thermal)] = 0
        adj_gis[torch.isnan(adj_gis)] = 0
        
        # Graph Convolution
        spectra = F.sigmoid(self._gc_spectra(spectra, adj_spectra))
        thermal = F.sigmoid(self._gc_thermal(thermal, adj_thermal))
        gis = F.sigmoid(self._gc_gis(gis, adj_gis))
        
        # feature-level data fusion and MLP learning
        x = torch.cat((spectra, thermal, gis), dim = 1)
        output = self._classifier(x)
        return output

    def save_state_to_file(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load_state_from_file(self, filepath):
        self.load_state_dict(torch.load(filepath))