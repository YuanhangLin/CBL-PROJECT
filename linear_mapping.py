import numpy as np
import os
from scipy.io import savemat, loadmat
from itertools import permutations
from glob import glob
import time
import pandas as pd

def pixmat(train_spectra, train_polygons, train_coordinates,
           test_spectra, test_polygons, test_coordinates):
    """
    This function performs pixel-matching linear mapping.
    Only the pixels that are included in both two dates can be used for constructing pairs.
    
    Inputs:
    train_spectra       : N_train x 174 Numpy array, spectra of pixels of training date
                          N_train : number of pixels of training date
                          174 : number of good bands (waterbands and zero bands are removed)
    train_polygon_names : N_train x 1 Numpy array, name of the polygon that a pixel belongs to 
    train_coordinates   : N_train x 2 Numpy array, relative coordinates (X,Y) of pixels in the HSI
    test_spectra        : N_test x 174 Numpy array, spectra of pixels of testing date
                          N_test : number of pixels of testing date 
                          174 : number of good bands (waterbands and zero bands are removed)
    test_polygon_names  : N_test x 1 Numpy array, name of the polygon that a pixel belongs to 
    test_coordinates    : N_test x 2 Numpy array, relative coordinates (X,Y) of pixels in the HSI
    
    Outputs:
    A                   : 174 x 174 Numpy array, linear mapping matrix that maps spectra of training date
                          to spectra of testing date
    """
    # Step 1 : find all the polygons that are included in both training and testing date
    common_polygons = list(set(train_polygons) & set(test_polygons))
    X_train = []
    X_test = []

    # Step 2 : iterate all common polygons and construct pairs by checking relative coordinates (X, Y) of the pixels
    for common_polygon in common_polygons:
        train_indices = np.where(train_polygons == common_polygon)[0]
        test_indices = np.where(test_polygons == common_polygon)[0]
        train_pos = train_coordinates[train_indices, :].tolist()
        test_pos = test_coordinates[test_indices, :].tolist()
        # For each pixel, if it is included in both training and testing date, use the spectra of two dates to construct pair
        # Otherwise, discard it, i.e., not all training pixels can be used to construct pairs
        for i in range(len(test_pos)):
            try:
                idx = train_pos.index(test_pos[i])
                X_train.append(train_spectra[train_indices[idx], :])
                X_test.append(test_spectra[test_indices[i], :])
            except ValueError:
                continue
    X_train = np.vstack(X_train).astype('float')
    X_test = np.vstack(X_test).astype('float')
    A = np.linalg.lstsq(X_test, X_train)[0] 
    return A

def randmat(train_spectra, train_polygons, test_spectra, test_polygons):
    """
    This function performs random-matching linear mapping by shuffling.
    One pixel can be used at most once to construct pairs.
    
    Inputs:
    train_spectra       : N_train x 174 Numpy array, spectra of pixels of training date
                          N_train : number of pixels of training date
                          174 : number of good bands(excluding waterbands and zero bands)
    train_polygon_names : N_train x 1 Numpy array, name of the polygon that a pixel belongs to 
    test_spectra        : N_test x 174 Numpy array, spectra of pixels of testing date
                          N_test : number of pixels of testing date 
    test_polygon_names  : N_test x 1 Numpy array, name of the polygon that a pixel belongs to 
    
    Outputs:
    A                   : 174 x 174 Numpy array, linear mapping matrix that maps spectra of training date
                          to spectra of testing date
    """
    # Step 1 : find all the polygons that are included in both training and testing date
    common_polygons = list(set(train_polygons) & set(test_polygons))
    X_train = []
    X_test = []
     # Step 2 : iterate all the common polygons and construct pairs by random shuffling
    for common_polygon in common_polygons:
        train_indices = np.where(train_polygons == common_polygon)[0]
        test_indices = np.where(test_polygons == common_polygon)[0]
        train_pixels = train_spectra[train_indices, :]
        test_pixels = test_spectra[test_indices, :]
        # Not all training pixels are guaranteed to be used for constructing pairs 
        min_length = min(len(train_pixels), len(test_pixels))
        perm_train = np.random.permutation(len(train_indices))
        perm_train = perm_train[:min_length]
        train_pixels = train_pixels[perm_train, :]
        perm_test = np.random.permutation(len(test_indices))
        perm_test = perm_test[:min_length]
        test_pixels = test_pixels[perm_test, :]
        X_train.append(train_pixels)
        X_test.append(test_pixels)
    X_train = np.vstack(X_train).astype('float')
    X_test = np.vstack(X_test).astype('float')
    A = np.linalg.lstsq(X_test, X_train)[0] 
    return A

def randmat_ver2(train_spectra, train_polygons, test_spectra, test_polygons):
    """
    This function performs random-matching linear mapping by randomly selection.
    One pixel can be used for MULTIPLE times to construct pairs.
    
    Inputs:
    train_spectra       : N_train x 174 Numpy array, spectra of pixels of training date
                          N_train : number of pixels of training date
                          174 : number of good bands(excluding waterbands and zero bands)
    train_polygon_names : N_train x 1 Numpy array, name of the polygon that a pixel belongs to 
    test_spectra        : N_test x 174 Numpy array, spectra of pixels of testing date
                          N_test : number of pixels of testing date 
    test_polygon_names  : N_test x 1 Numpy array, name of the polygon that a pixel belongs to 
    
    Outputs:
    A                   : 174 x 174 Numpy array, linear mapping matrix that maps spectra of training date
                          to spectra of testing date
    """
    # Step 1 : find all the polygons that are included in both training and testing date
    common_polygons = list(set(train_polygons) & set(test_polygons))
    X_train = []
    X_test = []
     # Step 2 : iterate all the common polygons and construct pairs by randomly selection
    for common_polygon in common_polygons:
        train_indices = np.where(train_polygons == common_polygon)[0]
        test_indices = np.where(test_polygons == common_polygon)[0]
        train_pixels = train_spectra[train_indices, :]
        test_pixels = test_spectra[test_indices, :]
        for i in range(len(train_pixels)):
            # Note: a pixel may be used multiple times to construct pairs
            j = np.random.randint(0, len(test_pixels))
            X_train.append(train_pixels[i, :])
            X_test.append(test_pixels[j, :])
    X_train = np.vstack(X_train).astype('float')
    X_test = np.vstack(X_test).astype('float')
    A = np.linalg.lstsq(X_test, X_train)[0] 
    return A

def get_date(file_name):
    """
    Input:
    file_name     : Python built-in string, raw data csv file
                    Below are the legal inputs:
                    *_AVIRIS_speclib_subset_spectra.csv
                    *_AVIRIS_speclib_subset_metadata.csv
                    *_AVIRIS_speclib_subset_trainval.csv
                    where * is the date
    Output:
    date          : Python built-in string
    """
    date = file_name.split('_')[0]
    return date

def get_spectra_and_polygon_name(file_name, bbl):
    """
    Inputs:
    file_name     : Python built-in string, raw data csv file
                    Must be *_AVIRIS_speclib_subset_spectra.csv, where * is date
    bbl           : 224 x 1 Numpy array
    Outputs:
    spectra       : Npixel x 174 Numpy array
    polygon_names : Npixel x 1 Numpy array
    """
    data = pd.read_csv(file_name)
    polygon_names = data['PolygonName'].values
    spectra = data.values[:,5:]
    spectra = spectra[:, bbl == 1] # remove water bands, 224 to 176
    spectra = spectra/10000
    spectra = spectra[:, 2:] # first two bands are zero bands, 176 to 174
    spectra[spectra < 0] = 0 
    spectra[spectra > 1] = 1 
    return spectra, polygon_names

def get_coordinates(file_name):
    """
    Inputs:
    file_name     : Python built-in string, raw data csv file
                    Must be *_AVIRIS_speclib_subset_metadata.csv, where * is date
    Outputs:
    coordinates   : Npixel x 2 Numpy array
    """
    coordinates = pd.read_csv(file_name)[['X','Y']].values
    return coordinates

def split_dataset(file_name, polygon_names, index):
    """
    Inputs:
    file_name     : Python built-in string, raw data csv file
                    Must be *_AVIRIS_speclib_subset_trainval.csv, where * is date
    polygon_names : Npixel x 1 Numpy array
    index         : Python built-in integer in range [0, 49]

    Outputs:
    split         : Npixel x 1 Numpy array
                    0 for testing, 1 for training, 2 for validation
    """
    assert(type(index) is int and index >= 0 and index <= 49)
    split = pd.read_csv(file_name).values
    split = split[:, index]
    train_pixels_indices = []
    valid_pixel_indices = []
    test_pixels_indices = []
    unique_polygons = np.unique(polygon_names)
    for unique_polygon in unique_polygons:
        temp_indices = np.where(polygon_names == unique_polygon)[0]
        if np.all(split[temp_indices] == 0):
            # if all pixels of a polygon are labeled 0, this polygon is a testing polygon
            # all of its pixels are used for testing
            test_pixels_indices += temp_indices.tolist()
        elif np.all(split[temp_indices] == 1):
            # if all pixels of a polygon are labeled 1, this polygon is a training polygon
            # if this polygon has no more than 10 pixels, all of its pixels are used for training
            train_pixels_indices += temp_indices.tolist()
        else:
            # if a polygon is selected as training polygon but has more than 10 pixels
            # all the pixels labeled 0 are used for VALIDATION, they are going to be relabeled as 2 
            # while the pixels labeled 1 are used for TRAINING
            train_pixels_indices += temp_indices[np.where(split[temp_indices] == 1)[0]].tolist()
            valid_pixel_indices += temp_indices[np.where(split[temp_indices] == 0)[0]].tolist()
        split[train_pixels_indices] = 1 # 1 for training
        split[valid_pixel_indices] = 2 # 2 for validation
        split[test_pixels_indices] = 0 # 0 for testing
    return split

def get_cda_transformation_matrix(spectra, split, labels, reduced_dim):
    """
    Inputs:
    spectra       : Npixel x 174 Numpy array
    split         : Npixel x 1 Numpy array
                    0 for testing, 1 for training, 2 for validation
    labels        : Npixel x 1 Numpy array
    reduced_dim   : Python integer, at most (Nclass - 1)
                    Nclass: number of different classes

    Output:
    A             : 174 x reduced_dim Numpy array
    """
    clf = LinearDiscriminantAnalysis(n_components = reduced_dim)
    indices = split[np.where(split == 1)[0]]
    spectra = spectra[indices, :]
    labels = labels[indices, :].astype(np.int64)
    clf.fit(spectra, labels)
    return clf.scalings_

def get_linear_mapping_matrices_using_PixMat():
    dates = ["130411", "130606", "131125", 
             "140416", "140600", "140829", 
             "150416", "150602", "150824"]
    bbl = loadmat("auxiliary_info.mat", squeeze_me = True)["bbl"]
    pixmat_matrices = dict()
    for i in range(len(dates)):
        train_date = dates[i]
        train_spectra, train_polygons = get_spectra_and_polygon_name(train_date + "_AVIRIS_speclib_subset_spectra.csv", bbl)
        for j in range(len(dates)):
            if j == i : continue
            test_date = dates[j]
            test_spectra, test_polygons = get_spectra_and_polygon_name(test_date + "_AVIRIS_speclib_subset_spectra.csv", bbl)
            