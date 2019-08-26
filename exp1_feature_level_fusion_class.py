#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path
import errno
import numpy as np
from PIL import Image

import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torchExp1Dataset
import torch.nn as nn
import torch.utils.data as tdata
from scipy.io import loadmat
from torchvision import datasets, transforms

class Exp1Dataset(tdata.Dataset):
    
    def __init__(self, root = "../data/", date = "130411", state = 0, split_index = 0):
        """
        state : 0 means testing, 1 means training, 2 means valid
        """
        file_name = root + "exp1_" + date + "_aggregated_dataset.mat"
        x = loadmat(file_name, squeeze_me = True)
        self._indices = np.where(x["split_indices"][split_index] == state)[0]
        self._polygon_names = x["unique_polygons"][self._indices][:, np.newaxis].astype(str)
        self._spectra = x["polygon_spectra"][self._indices]
        self._thermal = x["polygon_thermal"][self._indices]
        self._gis = x["polygon_gis"][self._indices]
        self._labels = x["polygon_labels"][self._indices][:, np.newaxis].astype(np.int)
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
               torch.from_numpy(self._labels[index]), \
               torch.from_numpy(self._missing_flag[index,:]).float()
               
               
class Exp1MLP(nn.Module):
    
    def __init__(self):
        
        super(Exp1MLP, self).__init__()
        self._spectra_fc1 = nn.Linear(174, 44)
        self._spectra_fc2 = nn.Linear(44, 11)
        self._thermal_fc = nn.Linear(5, 3)
        self._gis_fc = nn.Linear(10, 5)
        self._classifier = nn.Linear(19, 27)
        self._tanh = nn.Tanh()
    
    def forward(self, x):
        spectra = self._tanh(self._spectra_fc1(x[0]))
        spectra = self._spectra_fc2(spectra)
        thermal = self._tanh(self._thermal_fc(x[1]))
        gis = self._tanh(self._gis_fc(x[2]))
        out = torch.cat((spectra, thermal, gis), 1)
        out = self._classifier(out)
        return out
    
    def save_state_to_file(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load_state_from_file(self, filepath):
        self.load_state_dict(torch.load(filepath))