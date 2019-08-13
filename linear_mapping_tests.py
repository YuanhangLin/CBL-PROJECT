import unittest
from linear_mapping import *
import numpy as np
import os
from scipy.io import loadmat

class TestUnits(unittest.TestCase):

    def test_get_date(self):
        os.chdir("../data/")
        date = get_date("130411_AVIRIS_speclib_subset_metadata.csv")
        self.assertEqual("130411", date)
        date = get_date("130606_AVIRIS_speclib_subset_trainval.csv")
        self.assertEqual("130606", date)
        date = get_date("131125_AVIRIS_speclib_subset_spectra.csv.csv")
        self.assertEqual("131125", date)

    def test_get_spectra_and_polygon_name(self):
        os.chdir("../data/")
        bbl = loadmat("auxiliary_info.mat", squeeze_me = True)["bbl"]
        spectra, polygon_names = get_spectra_and_polygon_name("131125_AVIRIS_speclib_subset_spectra.csv", bbl)
        self.assertEqual(spectra.shape[1], 174)
        self.assertTrue(np.all(spectra <= 1))
        self.assertTrue(np.all(spectra >= 0))

    def test_get_coordinates(self):
        os.chdir("../data/")
        coordinates = get_coordinates("131125_AVIRIS_speclib_subset_metadata.csv")
        self.assertTrue(np.all(coordinates >= 0))
    
    def test_pixmat(self):
        os.chdir("../data/")
        bbl = loadmat("auxiliary_info.mat", squeeze_me = True)["bbl"]
        train_spectra, train_polygons = get_spectra_and_polygon_name("131125_AVIRIS_speclib_subset_spectra.csv", bbl)
        train_coordinates = get_coordinates("140416_AVIRIS_speclib_subset_metadata.csv")
        test_spectra, test_polygons = get_spectra_and_polygon_name("140416_AVIRIS_speclib_subset_spectra.csv", bbl)
        test_coordinates = get_coordinates("140416_AVIRIS_speclib_subset_metadata.csv")
        A = pixmat(train_spectra, train_polygons, train_coordinates,test_spectra, test_polygons, test_coordinates)
        self.assertTrue(A.shape[0], 174)
        self.assertTrue(A.shape[1], 174)
        print(A)

    def test_randmat(self):
        os.chdir("../data/")
        bbl = loadmat("auxiliary_info.mat", squeeze_me = True)["bbl"]
        train_spectra, train_polygons = get_spectra_and_polygon_name("131125_AVIRIS_speclib_subset_spectra.csv", bbl)
        train_coordinates = get_coordinates("140416_AVIRIS_speclib_subset_metadata.csv")
        test_spectra, test_polygons = get_spectra_and_polygon_name("140416_AVIRIS_speclib_subset_spectra.csv", bbl)
        test_coordinates = get_coordinates("140416_AVIRIS_speclib_subset_metadata.csv")
        A = randmat(train_spectra, train_polygons, test_spectra, test_polygons)
        self.assertTrue(A.shape[0], 174)
        self.assertTrue(A.shape[1], 174)
        print(A)
    
    def test_split_dataset(self):
        os.chdir("../data/")
        bbl = loadmat("auxiliary_info.mat", squeeze_me = True)["bbl"]
        _, polygons_names = get_spectra_and_polygon_name("131125_AVIRIS_speclib_subset_spectra.csv", bbl)
        split = split_dataset("131125_AVIRIS_speclib_subset_trainval.csv", polygons_names, 0)
        self.assertTrue(len(np.where(split == 0)[0]) > 0)
        self.assertTrue(len(np.where(split == 0)[0]) > 0)
        self.assertTrue(len(np.where(split == 0)[0]) > 0)


if __name__ == "__main__":
    unittest.main()