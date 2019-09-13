import random
import numpy as np
# from cvxopt import matrix, solvers
import math
# import matplotlib.pyplot as plt
from label_propagation_source_code_2008_paper import LocallyLinearEmbedding

class LNP:

    def __init__(self):

        self.X = []               # data
        self.Y = []               # labels for each samples
        self.true_labels = []

        self.attr_num = 2         # feature_num
        self.label_num = 2        # num of classes
        self.sigma = 5            # sigma for graph construction
        self.alpha = 0.99         # fraction of label information that xi receives from its neighbors
        self.maxk = 20             # number of nearest neighbors
        self.sub = []             # the max and min for each feature used for normalization
        self.gram = []            # Gram matrix
        self.W = []               # weight matrix (i.e., adjacency matrix for the samples)
        self.percentage_label_sample = 0.1 
        self.num_label_sample = 0 
        self.num_samples = 0

        self.neighbor = []        # knn 
        self.num = 0


    def get_x(self):
        return self.X

    def get_y(self):
        return self.Y

#    def build_graph(self):
#        self.max_min()
#        num_samples = len(self.X)
#
#        self.affinity_matrix =[[0 for col in range(self.num_samples)] for row in range(self.num_samples)]
#
#        for i in range(num_samples):
#            self.affinity_matrix[i][i] = [0.0, i]
#            for j in range(num_samples):
#                diff = 0.0
#                for k in range(self.attr_num):
#                    if i != j:
#                        dist = self.X[i][k] - self.X[j][k]
#                        dist = dist/self.sub[k]
#                        diff += dist ** 2
#
#                # self.gram[i][j] = diff       
#                if i != j:
#                    self.affinity_matrix[i][j] = [math.exp(diff/ (-2.0 * (self.sigma ** 2))), j]
#
#
#    def set_neighbor(self):
#        num_samples = len(self.X)
#
#        self.neighbor = [[]for row in range(num_samples)]
#        for i in range(num_samples):
#            temp = sorted(self.affinity_matrix[i], key=lambda x: x[0])
#            temp.reverse()
#            for k in range(self.maxk):
#                j = temp[k][1]
#                self.neighbor[i].append(j)
##            if i == 1:
##                print(self.X[i])
##                print(self.X[self.neighbor[i][0]])
##        print(self.neighbor)
#
#    def set_gram(self):
#        self.gram = [[0 for col in range(self.maxk)] for row in range(self.num_samples)]
#        for i in range(self.num_samples):
#            for j in range(self.maxk):
#                neighbor = self.neighbor[i][j]
#                diff = 0.0
#                for k in range(self.attr_num):
#                    dist = self.X[i][k] - self.X[neighbor][k]
#                    dist = dist / self.sub[k]
#                    diff += dist ** 2
#                self.gram[i][j] = diff 
##        print(self.gram)
#
#    def solve_weight(self):
#        self.W = LLE(self.X.T, self.maxk, 1)
#        
#
##    def solve_weight(self):
##        self.set_gram()
##        self.W = np.zeros((self.num_samples, self.num_samples), np.float32)
##        for i in range(self.num_samples):
##            self.cal_weight(i)
##        print("done")
##
##    def cal_weight(self, i):
##
##        '''
##        tempQ = np.zeros((self.maxk, self.maxk), np.double)
##        for j in range(self.maxk):
##            tempQ[j][j] = self.gram[i][j]
##        for m in range(self.maxk):
##            for n in range(self.maxk):
##                tempQ[m][n] = (self.gram[i][m] + self.gram[i][n]) / 2
##        print(type(tempQ))
##        Q = 2 * matrix(tempQ)
##        tempp = np.zeros((1, self.maxk), np.double)
##        p = matrix(tempp)                        # linear term
##        print(p)
##        G = matrix([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])        # GX + s = h, x > 0 
##        temph = np.zeros((1, self.maxk), np.double)
##        h = matrix(temph)
##        tempA = np.ones((1, self.maxk), np.double)
##        A = matrix(tempA)
##
##        '''
##
##        q11 = self.gram[i][0]  
##        q12 = (self.gram[i][0] + self.gram[i][1]) / 2
##        q13 = (self.gram[i][0] + self.gram[i][2]) / 2
##        q21 = (self.gram[i][0] + self.gram[i][1]) / 2
##        q22 = self.gram[i][1]
##        q23 = (self.gram[i][1] + self.gram[i][2]) / 2
##        q31 = (self.gram[i][0] + self.gram[i][2]) / 2
##        q32 = (self.gram[i][1] + self.gram[i][2]) / 2
##        q33 = self.gram[i][2]
##        Q = 2 * matrix([[q11, q21, q31], [q12, q22, q32], [q13, q23, q33]])
##        p = matrix([0.0, 0.0, 0.0])  
##        G = matrix([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  
##        h = matrix([0.0, 0.0, 0.0])
##        A = matrix([1.0, 1.0, 1.0], (1, 3))
##
##        b = matrix(1.0)                                                    # AX = b
##        sol = solvers.qp(Q, p, G, h, A, b)
##
###        print(sol['x'])
##        for j in range(self.maxk):
##            self.W[i][self.neighbor[i][j]] = sol['x'][j]


    def LNPiter(self):
        P = self.W.copy()

        tol = 0.00001     
        max_iter = 1000   
        self.num_label_sample = self.num_samples * self.percentage_label_sample
        num_unlabel_sample = self.num_samples - self.num_label_sample

        clamp_data_label = np.zeros((self.num_samples, self.label_num), np.float32)

        for i in range(int(self.num_label_sample)):
            # clamp_data_label[i] = self.Y[i]
            if self.Y[i] == -1:
                clamp_data_label[i][0] = 1
            else:
                clamp_data_label[i][1] = 1

        # for i in range(num_unlabel_sample):
            # clamp_data_label[i+self.num_label_sample] = 0


        # f = Xu
        label_function = clamp_data_label.copy()
        iter_num = 0
        pre_label_function = np.zeros((self.num_samples, self.label_num), np.float32)
        changed = np.abs(pre_label_function - label_function).sum()
        while iter_num < max_iter and changed > tol:
#            if iter_num % 1 == 0:
#                print("---> Iteration %d/%d, changed: %f" % (iter_num, max_iter, changed))
            pre_label_function = label_function
            iter_num += 1

            # propagation
            label_function = self.alpha * np.dot(P, label_function) + (1-self.alpha) * clamp_data_label

            # check converge
            changed = np.abs(pre_label_function - label_function).sum()

            # get terminate label of unlabeled data

        self.unlabel_data_labels = np.zeros(int(num_unlabel_sample))
        for i in range(int(num_unlabel_sample)):
            if label_function[i + int(self.num_label_sample)][0] > 0:
                self.unlabel_data_labels[i] = -1
            else:
                self.unlabel_data_labels[i] = 1
        
        '''
        correct_num = 0
        for i in xrange(num_unlabel_sample):
            if self.unlabel_data_labels[i] == self.Y[i + self.num_label_sample]:
                correct_num += 1
        print(self.unlabel_data_labels)
        accuracy = correct_num *100/ num_unlabel_sample
        print("Accuracy: %.2f%%" % accuracy)
        '''
        

    def rank_index(self):
        
#         A = 0.0
#         B = 0.0
#         C = 0.0
#         D = 0.0
#         numSamples = len(self.unlabel_data_labels)
#         for i in range(numSamples):
#             for j in range(i + 1, numSamples):
#                 if self.Y[i + int(self.num_label_sample)] == self.Y[j + int(self.num_label_sample)]:
#                     if self.unlabel_data_labels[i] == self.unlabel_data_labels[j]:
#                         A = A + 1
#                     else:
#                         B = B + 1
#                 else:
#                     if self.unlabel_data_labels[i] == self.unlabel_data_labels[j]:
#                         C = C + 1
#                     else:
#                         D = D + 1
# #        print(A, B, C, D)
#         accuracy = (A + D) / (A + B + C + D) * 100
#         return accuracy
        return np.where(self.unlabel_data_labels == self)

    def generate_data(self, unlabeled_percentage_ = 0.1, seed_ = 0):
        mu_1 = np.array([2, 2])
        sigma_1 = 0.01 * np.diag(np.ones(2))
        data_class_1 = np.random.multivariate_normal(mu_1, sigma_1, 100)
        labels_class_1 = np.ones(100).astype(int)

        mu_2 = np.array([-2, -2])
        sigma_2 = 0.1 * np.diag(np.ones(2))
        data_class_2 = np.random.multivariate_normal(mu_2, sigma_2, 100)
        labels_class_2 = -1*np.ones(100).astype(int)

        data = np.vstack((data_class_1, data_class_2))
        labels = np.concatenate((labels_class_1, labels_class_2))

        unlabeled_percentage = unlabeled_percentage_
        rng = np.random.RandomState(seed = seed_)
        unlabeled_idx = rng.rand(len(data)) < unlabeled_percentage

        partial_labels = labels.copy()
        partial_labels[unlabeled_idx] = 0

        idx = np.concatenate((np.where(partial_labels == 1)[0], np.where(partial_labels == -1)[0], np.where(partial_labels == 0)[0]))

        data = data[idx]
        labels = labels[idx]
        partial_labels[idx]

        self.X = data
        self.Y = partial_labels
        self.true_labels = labels
        self.num_samples = len(self.X)
        
        # plt.scatter(data[0:100,0], data[0:100,1], c = 'red')
        # plt.hold(True)
        # plt.scatter(data[100:,0], data[100:,1], c = 'blue')
        # plt.scatter(data[unlabeled_idx,0], data[unlabeled_idx, 1], c = 'black')
        # plt.show()
        
    def build_graph_using_LLE(self):
        my_LLE = LocallyLinearEmbedding(n_neighbors = 10, n_components = 2)
        my_LLE.fit(self.X)
        self.W = my_LLE.get_LLE_weight_matrix(self.X)


for i in range(1, 100, 5):
    test = LNP()
    test.generate_data(i/100, 0)    
    test.build_graph_using_LLE()
    test.LNPiter()
    accuracy = test.rank_index()
    print("Unlabeled percentage", i/100, "Accuracy: %d%%" %accuracy)