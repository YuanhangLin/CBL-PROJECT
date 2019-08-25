#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from utilities import *
from GraphConvNet import GraphConvNet

# parameters setting
num_epochs = 3000
learning_rate = 0.001
object_function = nn.CrossEntropyLoss(reduction = 'none')

# load data

# build adjs

# initialize net
net = GraphConvNet() 
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")