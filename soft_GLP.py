#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:52:04 2019

@author: linyuanhang
"""

import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

import sklearn

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.exceptions import ConvergenceWarning


class soft_GLP():
    """LabelSpreading model for semi-supervised learning

    This model is similar to the basic Label Propagation algorithm,
    but uses affinity matrix based on the normalized graph Laplacian
    and soft clamping across the labels.

    Read more in the :ref:`User Guide <label_propagation>`.

    Parameters
    ----------
    kernel : {'knn', 'rbf', callable}
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape [n_samples, n_features],
        and return a [n_samples, n_samples] shaped weight matrix

    gamma : float
      parameter for rbf kernel

    n_neighbors : integer > 0
      parameter for knn kernel

    alpha : float
      Clamping factor. A value in (0, 1) that specifies the relative amount
      that an instance should adopt the information from its neighbors as
      opposed to its initial label.
      alpha=0 means keeping the initial label information; alpha=1 means
      replacing all initial information.

    max_iter : integer
      maximum number of iterations allowed

    tol : float
      Convergence tolerance: threshold to consider the system at steady
      state

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        Input array.

    classes_ : array, shape = [n_classes]
        The distinct labels used in classifying instances.

    label_distributions_ : array, shape = [n_samples, n_classes]
        Categorical distribution for each item.

    transduction_ : array, shape = [n_samples]
        Label assigned to each item via the transduction.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import LabelSpreading
    >>> label_prop_model = LabelSpreading()
    >>> iris = datasets.load_iris()
    >>> rng = np.random.RandomState(42)
    >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> label_prop_model.fit(iris.data, labels)
    LabelSpreading(...)

    References
    ----------
    Dengyong Zhou, Olivier Bousquet, Thomas Navin Lal, Jason Weston,
    Bernhard Schoelkopf. Learning with local and global consistency (2004)
    http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.115.3219

    See Also
    --------
    LabelPropagation : Unregularized graph based semi-supervised learning
    """

    _variant = 'spreading'


    def __init__(self, kernel='rbf', gamma=20, n_neighbors=7,
                 alpha=0.99, max_iter=30, tol=1e-3, n_jobs=None):

        self.max_iter = max_iter
        self.tol = tol

        # kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors

        # clamping factor
        self.alpha = alpha

        self.n_jobs = n_jobs
        
    def _get_kernel(self, X, y=None):
        if self.kernel == "rbf":
            if y is None:
                return rbf_kernel(X, X, gamma=self.gamma)
            else:
                return rbf_kernel(X, y, gamma=self.gamma)
        elif self.kernel == "knn":
            if self.nn_fit is None:
                self.nn_fit = NearestNeighbors(self.n_neighbors,
                                               n_jobs=self.n_jobs).fit(X)
            if y is None:
                return self.nn_fit.kneighbors_graph(self.nn_fit._fit_X,
                                                    self.n_neighbors,
                                                    mode='connectivity')
            else:
                return self.nn_fit.kneighbors(y, return_distance=False)
        elif callable(self.kernel):
            if y is None:
                return self.kernel(X, X)
            else:
                return self.kernel(X, y)
        else:
            raise ValueError("%s is not a valid kernel. Only rbf and knn"
                             " or an explicit function "
                             " are supported at this time." % self.kernel)
        
    def _build_graph(self):
        """Graph matrix for Label Spreading computes the graph laplacian"""
        # compute affinity matrix (or gram matrix)
        if self.kernel == 'knn':
            self.nn_fit = None
        n_samples = self.X_.shape[0]
        affinity_matrix = self._get_kernel(self.X_)
        laplacian = csgraph.laplacian(affinity_matrix, normed=True)
        laplacian = -laplacian
        if sparse.isspmatrix(laplacian):
            diag_mask = (laplacian.row == laplacian.col)
            laplacian.data[diag_mask] = 0.0
        else:
            laplacian.flat[::n_samples + 1] = 0.0  # set diag to 0.0
        return laplacian
    
    def predict_proba(self, X):
        """Predict probability for each possible outcome.

        Compute the probability estimates for each single sample in X
        and each possible outcome seen during training (categorical
        distribution).

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]

        Returns
        -------
        probabilities : array, shape = [n_samples, n_classes]
            Normalized probability distributions across
            class labels
        """
        check_is_fitted(self, 'X_')

        X_2d = check_array(X, accept_sparse=['csc', 'csr', 'coo', 'dok',
                                             'bsr', 'lil', 'dia'])
        weight_matrices = self._get_kernel(self.X_, X_2d)
        if self.kernel == 'knn':
            probabilities = np.array([
                np.sum(self.label_distributions_[weight_matrix], axis=0)
                for weight_matrix in weight_matrices])
        else:
            weight_matrices = weight_matrices.T
            probabilities = np.dot(weight_matrices, self.label_distributions_)
        normalizer = np.atleast_2d(np.sum(probabilities, axis=1)).T
        probabilities /= normalizer
        return probabilities
    
    def predict(self, X):
        """Performs inductive inference across the model.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]

        Returns
        -------
        y : array_like, shape = [n_samples]
            Predictions for input data
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)].ravel()
    
    def fit(self, X, y):
        """Fit a semi-supervised label propagation model based

        All the input data is provided matrix X (labeled and unlabeled)
        and corresponding label matrix y with a dedicated marker value for
        unlabeled samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix will be created from this

        y : array_like, shape = [n_samples]
            n_labeled_samples (unlabeled points are marked as -1)
            All unlabeled samples will be transductively assigned labels

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = check_X_y(X, y)
        self.X_ = X
        check_classification_targets(y)

        # actual graph construction (implementations should override this)
        graph_matrix = self._build_graph()

        # label construction
        # construct a categorical distribution for classification only
        classes = np.unique(y)
        classes = (classes[classes != -1])
        self.classes_ = classes

        n_samples, n_classes = len(y), len(classes)

        alpha = self.alpha
        if self._variant == 'spreading' and \
                (alpha is None or alpha <= 0.0 or alpha >= 1.0):
            raise ValueError('alpha=%s is invalid: it must be inside '
                             'the open interval (0, 1)' % alpha)
        y = np.asarray(y)
        unlabeled = y == -1

        # initialize distributions
        self.label_distributions_ = np.zeros((n_samples, n_classes))
        for label in classes:
            self.label_distributions_[y == label, classes == label] = 1
        
         # for the unlabeled data, by default its row is a zero vector
       

        y_static = np.copy(self.label_distributions_)
        # LabelPropagation
        
        y_static[unlabeled] = 0

        l_previous = np.zeros((self.X_.shape[0], n_classes))

        unlabeled = unlabeled[:, np.newaxis]
        if sparse.isspmatrix(graph_matrix):
            graph_matrix = graph_matrix.tocsr()

        for self.n_iter_ in range(self.max_iter):
            if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
                break

            l_previous = self.label_distributions_
            self.label_distributions_ = safe_sparse_dot(
                graph_matrix, self.label_distributions_)

            if self._variant == 'propagation':
                normalizer = np.sum(
                    self.label_distributions_, axis=1)[:, np.newaxis]
                self.label_distributions_ /= normalizer
                self.label_distributions_ = np.where(unlabeled,
                                                     self.label_distributions_,
                                                     y_static)
            else:
                # clamp
                self.label_distributions_ = np.multiply(
                    alpha, self.label_distributions_) + y_static
        else:
            warnings.warn(
                'max_iter=%d was reached without convergence.' % self.max_iter,
                category=ConvergenceWarning
            )
            self.n_iter_ += 1

        normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
        self.label_distributions_ /= normalizer

        # set the transduction item
        transduction = self.classes_[np.argmax(self.label_distributions_,
                                               axis=1)]
        self.transduction_ = transduction.ravel()
        return self
    