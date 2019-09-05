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
        
        # MLPs for each sensor
        nhids_each_sensors = [math.floor(nhid[0]/2)+1, math.floor(nhid[1]/2)+1, math.floor(nhid[2]/2)+1]
        self._mlp_spectra = nn.Linear(nhid[0], nhids_each_sensors[0])
        self._mlp_thermal = nn.Linear(nhid[1], nhids_each_sensors[1])
        self._mlp_gis = nn.Linear(nhid[2], nhids_each_sensors[2])
        
        # MLPs for data-fusion and classification
        self._classifier = nn.Linear(sum(nhids_each_sensors), nclass)

        # a lot of people use it but not here...
        self.dropout = dropout

    def forward(self, x, adjs):
        spectra, thermal, gis = x[0], x[1], x[2]
        adj_spectra, adj_thermal, adj_gis = adjs[0], adjs[1], adjs[2]
        
        # remove all nan
        adj_spectra[torch.isnan(adj_spectra)] = 0
        adj_thermal[torch.isnan(adj_thermal)] = 0
        adj_gis[torch.isnan(adj_gis)] = 0
        
        # Graph Convolution
        spectra = self._gc_spectra(spectra, adj_spectra)
        thermal = self._gc_thermal(thermal, adj_thermal)
        gis = self._gc_gis(gis, adj_gis)
        
        # MLP for intermediate features
        spectra = F.tanh(self._mlp_spectra(spectra))
        thermal = F.tanh(self._mlp_thermal(thermal))
        gis = F.tanh(self._mlp_gis(gis))
        
        # feature-level data fusion and classification
        x = torch.cat((spectra, thermal, gis), dim = 1)
        output = self._classifier(x)
        return output

    def save_state_to_file(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load_state_from_file(self, filepath):
        self.load_state_dict(torch.load(filepath))