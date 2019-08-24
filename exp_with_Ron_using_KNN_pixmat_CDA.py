#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:15:59 2019

@author: linyuanhang
"""

import sklearn.tree, sklearn.ensemble
from utilities import *
from scipy.io import loadmat, savemat
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

"""
EXP1 : pixmat with KNN using Ron's split
"""

def pixmat_KNN_using_unbalanced_split(split = 0):
    """
    This function applies PixMat first, and then use KNN to classify.
    """
    train_date, test_date = "130411", "140416"
    A = pixmat_between_two_dates(train_date, test_date, path = "raw_file/")
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    trainset = loadmat("SusanSpectraProcessed" + train_date + ".mat", squeeze_me = True)
    testset = loadmat("SusanSpectraProcessed" + test_date + ".mat", squeeze_me = True)
    indices = trainset["train_indices_splitter"][split] - 1
    train_spectra = trainset["spectra"][:, trainset["bbl"] == 1]
    train_spectra = train_spectra[indices, 2:]
    train_targets = trainset["labels"][indices]
    KNN_classifier.fit(train_spectra, train_targets)
    
    test_spectra = testset["spectra"][:, testset["bbl"] == 1]
    test_spectra = test_spectra[:, 2:]
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    test_est = KNN_classifier.predict(test_spectra)
    print("pixmat_KNN_unbalance_split accuracy by pixel", np.where(test_targets == test_est)[0].shape[0] / len(test_targets))
    
def randmat_KNN_using_unbalanced_split(split = 0):
    """
    This function applies PixMat first, and then use KNN to classify.
    """
    train_date, test_date = "130411", "140416"
    A = randmat_between_two_dates(train_date, test_date, path = "raw_file/")
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    trainset = loadmat("SusanSpectraProcessed" + train_date + ".mat", squeeze_me = True)
    testset = loadmat("SusanSpectraProcessed" + test_date + ".mat", squeeze_me = True)
    indices = trainset["train_indices_splitter"][split] - 1
    train_spectra = trainset["spectra"][:, trainset["bbl"] == 1]
    train_spectra = train_spectra[indices, 2:]
    train_targets = trainset["labels"][indices]
    KNN_classifier.fit(train_spectra, train_targets)
    
    test_spectra = testset["spectra"][:, testset["bbl"] == 1]
    test_spectra = test_spectra[:, 2:]
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    test_est = KNN_classifier.predict(test_spectra)
    print("randmat_KNN_unbalance_split accuracy by pixel", np.where(test_targets == test_est)[0].shape[0] / len(test_targets))

"""
EXP2 : pixmat with KNN using Susan's split
"""

def pixmat_with_KNN_using_balanced_split(split = 0):
    train_date, test_date = "130411", "140416"
    A = pixmat_between_two_dates(train_date, test_date, path = "raw_file/")
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat("SusanSpectraProcessed" + train_date + "_classesremoved.mat", squeeze_me = True)
    testset = loadmat("SusanSpectraProcessed" + test_date + "_classesremoved.mat", squeeze_me = True)
    indices = trainset["train_indices_splitter"][split] 
    train_spectra = trainset["spectra"][indices,:]
    train_targets = trainset["labels"][indices]
    
    KNN_classifier.fit(train_spectra, train_targets)
    
    test_spectra = testset["spectra"]
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    test_est = KNN_classifier.predict(test_spectra)
    
    print("pixmat_KNN_balance_split accuracy by pixel:", np.where(test_targets == test_est)[0].shape[0] / len(test_targets))
    
def randmat_with_KNN_using_balanced_split(split = 0):
    train_date, test_date = "130411", "140416"
    A = randmat_between_two_dates(train_date, test_date, path = "raw_file/")
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat("SusanSpectraProcessed" + train_date + "_classesremoved.mat", squeeze_me = True)
    testset = loadmat("SusanSpectraProcessed" + test_date + "_classesremoved.mat", squeeze_me = True)
    indices = trainset["train_indices_splitter"][split] 
    train_spectra = trainset["spectra"][indices,:]
    train_targets = trainset["labels"][indices]
    
    KNN_classifier.fit(train_spectra, train_targets)
    
    test_spectra = testset["spectra"]
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    test_est = KNN_classifier.predict(test_spectra)
    
    print("randmat_KNN_balance_split accuracy by pixel:", np.where(test_targets == test_est)[0].shape[0] / len(test_targets))

"""
EXP3 : pixmat with CDA using Susan's split
"""
def pixmat_with_CDA_KNN_using_balanced_split(split = 0):
    train_date, test_date = "130411", "140416"
    A = pixmat_between_two_dates(train_date, test_date, path = "raw_file/")
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat("SusanSpectraProcessed" + train_date + "_classesremoved.mat", squeeze_me = True)
    testset = loadmat("SusanSpectraProcessed" + test_date + "_classesremoved.mat", squeeze_me = True)
    indices = trainset["train_indices_splitter"][split] 
    train_spectra = trainset["spectra"][indices,:]
    train_targets = trainset["labels"][indices]
    
    # CDA for dimensionality reduction
    clf = LinearDiscriminantAnalysis(n_components = 26)
    clf.fit(train_spectra, train_targets)
    train_spectra = clf.transform(train_spectra)
    
    KNN_classifier.fit(train_spectra, train_targets)
    
    test_spectra = testset["spectra"]
    # pixmat 
    test_spectra = np.dot(test_spectra, A)
    # CDA 
    test_spectra = clf.transform(test_spectra)
    test_targets = testset["labels"]
    test_est = KNN_classifier.predict(test_spectra)
    
    print("pixmat_with_CDA_KNN_balance_split accuracy by pixel:", np.where(test_targets == test_est)[0].shape[0] / len(test_targets))


def randmat_with_CDA_KNN_using_balanced_split(split = 0):
    train_date, test_date = "130411", "140416"
    A = randmat_between_two_dates(train_date, test_date, path = "raw_file/")
    KNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 30)
    
    trainset = loadmat("SusanSpectraProcessed" + train_date + "_classesremoved.mat", squeeze_me = True)
    testset = loadmat("SusanSpectraProcessed" + test_date + "_classesremoved.mat", squeeze_me = True)
    indices = trainset["train_indices_splitter"][split] 
    train_spectra = trainset["spectra"][indices,:]
    train_targets = trainset["labels"][indices]
    
    # CDA for dimensionality reduction
    clf = LinearDiscriminantAnalysis(n_components = 26)
    clf.fit(train_spectra, train_targets)
    train_spectra = clf.transform(train_spectra)
    
    KNN_classifier.fit(train_spectra, train_targets)
    
    test_spectra = testset["spectra"]
    # randmat 
    test_spectra = np.dot(test_spectra, A)
    # CDA 
    test_spectra = clf.transform(test_spectra)
    
    test_targets = testset["labels"]
    test_est = KNN_classifier.predict(test_spectra)
    
    print("randmat_with_CDA_KNN_balance_split accuracy by pixel:", np.where(test_targets == test_est)[0].shape[0] / len(test_targets))

def pixmat_with_CDA_decisiontree_using_balanced_split(split = 0):
    train_date, test_date = "130411", "140416"
    A = pixmat_between_two_dates(train_date, test_date, path = "raw_file/")
    decision_tree_classifier = sklearn.tree.DecisionTreeClassifier()

    trainset = loadmat("SusanSpectraProcessed" + train_date + "_classesremoved.mat", squeeze_me = True)
    testset = loadmat("SusanSpectraProcessed" + test_date + "_classesremoved.mat", squeeze_me = True)
    indices = trainset["train_indices_splitter"][split] 
    train_spectra = trainset["spectra"][indices,:]
    train_targets = trainset["labels"][indices]
    
    # CDA for dimensionality reduction
    clf = LinearDiscriminantAnalysis(n_components = 26)
    clf.fit(train_spectra, train_targets)
    train_spectra = clf.transform(train_spectra)
    
    decision_tree_classifier.fit(train_spectra, train_targets)
    
    test_spectra = testset["spectra"]
    # pixmat 
    test_spectra = np.dot(test_spectra, A)
    # CDA 
    test_spectra = clf.transform(test_spectra)
    test_targets = testset["labels"]
    test_est = decision_tree_classifier.predict(test_spectra)
    
    print("pixmat_with_CDA_decisiontree_balance_split accuracy by pixel:", np.where(test_targets == test_est)[0].shape[0] / len(test_targets))

def pixmat_with_CDA_randomforest_using_balanced_split(split = 0):
    train_date, test_date = "130411", "140416"
    A = pixmat_between_two_dates(train_date, test_date, path = "raw_file/")
    RandomForestClassifier = sklearn.ensemble.RandomForestClassifier()

    trainset = loadmat("SusanSpectraProcessed" + train_date + "_classesremoved.mat", squeeze_me = True)
    testset = loadmat("SusanSpectraProcessed" + test_date + "_classesremoved.mat", squeeze_me = True)
    indices = trainset["train_indices_splitter"][split] 
    train_spectra = trainset["spectra"][indices,:]
    train_targets = trainset["labels"][indices]
    
    # CDA for dimensionality reduction
    clf = LinearDiscriminantAnalysis(n_components = 26)
    clf.fit(train_spectra, train_targets)
    train_spectra = clf.transform(train_spectra)
    
    RandomForestClassifier.fit(train_spectra, train_targets)
    
    test_spectra = testset["spectra"]
    # pixmat 
    test_spectra = np.dot(test_spectra, A)
    # CDA 
    test_spectra = clf.transform(test_spectra)
    test_targets = testset["labels"]
    test_est = RandomForestClassifier.predict(test_spectra)
    
    print("pixmat_with_CDA_randomforest_balance_split accuracy by pixel:", np.where(test_targets == test_est)[0].shape[0] / len(test_targets))

def pixmat_with_CDA_randomforest_using_unbalanced_split(split = 0):
    train_date, test_date = "130411", "140416"
    A = pixmat_between_two_dates(train_date, test_date, path = "raw_file/")
    RandomForestClassifier = sklearn.ensemble.RandomForestClassifier()
    
    trainset = loadmat("SusanSpectraProcessed" + train_date + ".mat", squeeze_me = True)
    testset = loadmat("SusanSpectraProcessed" + test_date + ".mat", squeeze_me = True)
    indices = trainset["train_indices_splitter"][split] - 1
    train_spectra = trainset["spectra"][:, trainset["bbl"] == 1]
    train_spectra = train_spectra[indices, 2:]
    train_targets = trainset["labels"][indices]
    RandomForestClassifier.fit(train_spectra, train_targets)
    
    test_spectra = testset["spectra"][:, testset["bbl"] == 1]
    test_spectra = test_spectra[:, 2:]
    # pixmat 
    test_spectra = np.dot(test_spectra, A)
    test_targets = testset["labels"]
    test_est = RandomForestClassifier.predict(test_spectra)
    
    RandomForestClassifier.fit(train_spectra, train_targets)
    
    # CDA 
    test_spectra = clf.transform(test_spectra)
    test_targets = testset["labels"]
    test_est = RandomForestClassifier.predict(test_spectra)
    
    print("pixmat_with_CDA_randomforest_balance_split accuracy by pixel:", np.where(test_targets == test_est)[0].shape[0] / len(test_targets))


if __name__ == "__main__":
    for i in range(10):
        # EXP1 
#        pixmat_with_KNN_using_unbalanced_split(split = i)
#        randmat_with_KNN_using_unbalanced_split(split = i)
        # EXP2 
#        pixmat_with_KNN_using_balanced_split(split = i)
#        randmat_with_KNN_using_balanced_split(split = i)
#        pixmat_with_CDA_KNN_using_balanced_split(split = i)
#        randmat_with_CDA_KNN_using_balanced_split(split = i)
#        pixmat_with_CDA_decisiontree_using_balanced_split(split = i)
        
        pixmat_with_CDA_randomforest_using_unbalanced_split(split = i)