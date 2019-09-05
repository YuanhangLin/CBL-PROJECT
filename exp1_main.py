#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import numpy as np
import torch.utils.data as tdata
from scipy.io import loadmat
from torchvision import datasets, transforms

from utilities import *

class Exp1Dataset(tdata.Dataset):
    
    def __init__(self, root = "../data/", date = "130411", state = 0, split_index = 0):
        """
        state : 0 means testing, 1 means training, 2 means valid
        """
        file_name = root + "exp1_" + date + "_aggregated_dataset_exp1.mat"
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
        self._spectra_fc1 = nn.Linear(174, 174)
        self._spectra_fc2 = nn.Linear(174, 11)
        self._thermal_fc = nn.Linear(5, 3)
        self._gis_fc = nn.Linear(10, 5)
        self._classifier = nn.Linear(19, 27)
        self._spectra_fc1.weight = nn.Parameter(torch.diag(torch.diag(self._spectra_fc1.weight)))
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
    

def make_dataset_aggregated_polygon(date):
    os.chdir("../data/")
    ### get bbl info
    bbl, label_name_mapping = get_auxiliary_info("auxiliary_info.mat")
    spectra_raw_csv = date + "_AVIRIS_speclib_subset_spectra.csv"
    ### pixel_level_spectra preprocess
    _, polygon_names = get_spectra_and_polygon_name(spectra_raw_csv, bbl)
    ### get list of unique polygons
    unique_polygons = np.unique(polygon_names)[:, np.newaxis].astype(str)
    ### aggregate spectra
    polygon_spectra = get_spectra(spectra_raw_csv, bbl, unique_polygons)
    ### aggregate thermal
    thermal_raw_csv = date + "_AVIRIS+MASTER_spectral_library_spectra.csv"
    polygon_thermal = get_thermal(thermal_raw_csv, unique_polygons)
    ### aggregate gis features
    gis_raw_csv = "topo_variables_spectral_library_spectra.csv"
    polygon_gis = get_gis_feature(gis_raw_csv, unique_polygons)
    ### split by polygon
    split_indices_csv = date + "_AVIRIS_speclib_subset_trainval.csv"
    split_indices = []
    for i in range(20):
        indices = split_dataset_for_aggregated_polygon(split_indices_csv, polygon_names, unique_polygons, 0)
        split_indices.append(indices)
    split_indices = np.array(split_indices)
    ### assign labels for each labels
    polygon_labels = assign_labels_for_polygons(label_name_mapping, unique_polygons)
    data = [polygon_spectra, polygon_thermal, polygon_gis]
    feature_mean = [None] * 3
    missing_data_flag = [None] * 3
    for i in range(len(data)):
        feature_mean[i], missing_data_flag[i] = get_statistic(data[i])
        data[i] = replace_missing_data(data[i], feature_mean[i], missing_data_flag[i])
    missing_data_flag = np.array(missing_data_flag).squeeze().T
    savemat("exp1_" + date + "_aggregated_dataset_exp1.mat", { "unique_polygons":unique_polygons, 
            "polygon_spectra":polygon_spectra, "polygon_thermal":polygon_thermal, 
            "polygon_gis": polygon_gis, "split_indices":split_indices,
            "polygon_labels" : polygon_labels, 
            "missing_data_flag" : missing_data_flag, 
            "spectra_mean": feature_mean[0], "thermal_mean" : feature_mean[1], "gis_mean" : feature_mean[2],
            "spectra_missing_flag" : missing_data_flag[0], "thermal_missing_flag" : missing_data_flag[1], 
            "gis_missing_flag" : missing_data_flag[2]})            


def make_all_date_dataset():
    dates = ["130411", "130606", "131125", "140416", "140600", "140829", "150416", "150602", "150824"]
    for date in dates:
        make_dataset_aggregated_polygon(date)

def exp1():
    make_dataset_aggregated_polygon("130411")
    learning_rate = 0.001
    num_epochs = 3000
    exp1_net = Exp1MLP()
    exp1_net.cuda()
    criterion = nn.CrossEntropyLoss(reduction = 'none') # calculate the loss for each training instance
    optimizer = torch.optim.Adam(exp1_net.parameters(), lr = learning_rate)
    exp1_net.train()            
    trainset = Exp1Dataset(state = 1)
    testset = Exp1Dataset(state = 0)
    validset = Exp1Dataset(state = 2)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32, shuffle = True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    average_loss, ctr = 0, 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            spectra, thermal, gis, labels, missing_flag = data[0].to(device), data[1].to(device), \
                                                          data[2].to(device), data[3].to(device), \
                                                          data[4].to(device)
            spectra, thermal, gis, labels, missing_flag = torch.autograd.Variable(spectra), torch.autograd.Variable(thermal), \
                                                          torch.autograd.Variable(gis), torch.autograd.Variable(labels), \
                                                          torch.autograd.Variable(missing_flag)                          
            optimizer.zero_grad()
            x = [spectra, thermal, gis]
            outputs = exp1_net(x)
            loss = criterion(outputs, labels.squeeze(0) if outputs.shape[0] == 1 else labels.squeeze())
            if loss.item() > 5 :
                print("bad training instance")
                continue
            loss.backward()
            if missing_flag[0][0] == 1:
                exp1_net._spectra_fc1.zero_grad()
                exp1_net._spectra_fc2.zero_grad()
                exp1_net._classifier.zero_grad()
            if missing_flag[0][1] == 1:
                exp1_net._thermal_fc.zero_grad()
                exp1_net._classifier.zero_grad()
            if missing_flag[0][2] == 1:
                exp1_net._gis_fc.zero_grad()
                exp1_net._classifier.zero_grad()
            optimizer.step()
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            _, predicted = torch.max(outputs, 1)

            running_loss = 0.0
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            spectra, thermal, gis, labels, missing_flag = data[0].to(device), data[1].to(device), \
                                                          data[2].to(device), data[3].to(device), \
                                                          data[4].to(device)
            spectra, thermal, gis, labels, missing_flag = torch.autograd.Variable(spectra), torch.autograd.Variable(thermal), \
                                                          torch.autograd.Variable(gis), torch.autograd.Variable(labels), \
                                                          torch.autograd.Variable(missing_flag)
            x = [spectra, thermal, gis]
            outputs = exp1_net(x)
            _, predicted = torch.max(outputs.data, 1)
            labels = labels.squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the test sets: %d %%' % (
        100 * correct / total))
        
if __name__ == "__main__":
    exp1()