#! coding: utf-8
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import re
import copy
from sklearn import datasets
from sklearn.decomposition import PCA

class PCA_CLASS:
    def __init__(self):
        self.nrow = 0
        self.ncol = 0
        self.data_flg = 1
        self.data_sets = self.getDataSets()
        # self.eigenvalue()
        # self.pca = self.calc_pca()
        # self.plot_feature()
        print(self.numpy_pca())

    def plot_feature(self):
        for feature in self.data_sets['data']:
            plt.plot(feature[0], feature[1], "o")
        plt.show()

    def covariance(self, x, y):
        mean_x = sum(self.data_sets['data'][:,x])/self.nrow
        mean_y = sum(self.data_sets['data'][:,y])/self.nrow
        return sum([(self.data_sets['data'][i][x]-mean_x)*(self.data_sets['data'][i][y]-mean_y)
            for i in range(self.nrow)])/self.nrow

    def variance_covariance_matrix(self):
        matrix = np.ones((self.ncol, self.ncol))
        for i in range(self.ncol):
            for j in range(self.ncol):
                matrix[i][j] = self.covariance(i,j)
        return matrix

    def trace(self, matrix):
        sum([matrix[i][i] for i in range(self.ncol)])

    def calc_b(self, matrix, x):
        b = [0]*self.ncol
        for i in range(self.ncol):
            for j in range(self.ncol):
                b[i] += matrix[i][j]*x[j]
        return b

    def jacobi(self, matrix):
        # matrix = [[3.0,-6.0,9.0],[2.0,5.0,-8.0],[1.0,-4.0,7.0]]
        # matrix = [[1.0,7.0,3.0],[7.0,4.0,-5.0],[3.0,-5.0,6.0]]
        # self.ncol = 3

        b = [0 for i in range(self.ncol)]

        # print(matrix)
        count = 0
        v = np.array([[1.0 if i == j else 0.0 for i in range(self.ncol)] for j in range(self.ncol)])
        a = np.array(copy.deepcopy(matrix))
        while True:
            i = 0
            j = 0
            x = 0.0
            for ia in range(self.ncol):
                for ja in range(self.ncol):
                    if ia != ja and abs(a[ia][ja]) > x:
                        i = ia
                        j = ja
                        x = abs(a[ia][ja])
            if not x: break
            aii = a[i][i]
            aij = a[i][j]
            ajj = a[j][j]

            alpha = (aii-ajj)/2.0
            beta = -aij
            gamma = abs(alpha)/np.sqrt(alpha**2 + beta**2)
            # print(i, j, beta, np.sqrt(alpha*alpha+aij*aij))

            st = np.sqrt((1-gamma)/2)
            ct = np.sqrt((1+gamma)/2)
            if alpha*beta < 0: st = -st

            for m in range(self.ncol):
                tmp = ct*a[i][m] - st*a[j][m]
                a[j][m] = st*a[i][m] + ct*a[j][m]
                a[i][m] = tmp

            for m in range(self.ncol):
                a[m][i] = a[i][m]
                a[m][j] = a[j][m]

            a[i][i] = ct*ct*aii + st*st*ajj - 2*st*ct*aij
            a[i][j] = st*ct*(aii-ajj) + (ct**2 - st**2)*aij
            a[j][i] = st*ct*(aii-ajj) + (ct**2 - st**2)*aij
            a[j][j] = st*st*aii + ct*ct*ajj + 2*st*ct*aij

            for m in range(self.ncol):
                tmp = ct*v[m][i] - st*v[m][j]
                v[m][j] = st*v[m][i] + ct*v[m][j]
                v[m][i] = tmp

            e = 0.0
            for ia in range(self.ncol):
                for ja in range(self.ncol):
                    if ia != ja:
                        e += abs(a[ia][ja])

            if e < 1e-7: break

        # print(count)
        # print(v)
        # print(a)
        value  = []
        for i in range(self.ncol):
            for j in range(self.ncol):
                if i == j:
                    value.append(a[i][j])
        vector = v
        # print(value)
        return (value, vector)

    def QR(self, matrix):
        print

    def swap(self, x, y):
        tmp = x
        x = y
        y = tmp

    def vector_sort(self, eigvalue, eigvector):
        self.data_sets['data'] = self.data_sets['data'].T
        # print("swap value: ", eigvalue)
        # print("swap vevct: ", eigvector)
        for i in range(self.ncol):
            for j in range(self.ncol):
                if i != j and eigvalue[i] > eigvalue[j]:
                    # print(i, j, eigvalue[i], eigvalue[i])
                    eigvalue[i], eigvalue[j] = eigvalue[j], eigvalue[i]
                    eigvector[i], eigvector[j] = copy.deepcopy(eigvector[j]), copy.deepcopy(eigvector[i])
                    self.data_sets['data'][i], self.data_sets['data'][j] = copy.deepcopy(self.data_sets['data'][j]), copy.deepcopy(self.data_sets['data'][i])
        self.data_sets['data'] = self.data_sets['data'].T

    def calc_pca(self):
        matrix = self.variance_covariance_matrix()
        iris_bar = np.array([row - np.mean(row) for row in self.data_sets['data'].T]).T
        # print(matrix)

        # matrix = np.array([[1.0,7.0,3.0],[7.0,4.0,-5.0],[3.0,-5.0,6.0]])

        # self.jacobi(matrix)
        (jeigvalue, jeigvector) = self.jacobi(matrix)
        jeigvector = jeigvector.T

        (eigvalue, eigvector) = np.linalg.eig(matrix)
        eigvector = eigvector.T

        self.vector_sort( eigvalue,  eigvector)
        self.vector_sort(jeigvalue, jeigvector)
        # print("jeigen value", jeigvalue)
        # print("jeigen vector", jeigvector)
        # print()

        # print("eigen value", eigvalue)
        # print("eigen vector", eigvector)
        # print()

        dim = 2
        # components = eigvector[:dim,]
        components = jeigvector[:dim,]
        # iris_bar = データから平均ベクトルを引いたデータ
        print()
        m = np.dot(iris_bar, components.T)
        print(m)
        print()

    def numpy_pca(self):
        # 主成分分析による次元削減
        loaded = datasets.load_iris()
        pca = PCA(n_components=2)
        X = self.data_sets.data
        pca.fit(X)
        X_pca= pca.transform(X)
        # 主成分分析後のサイズ
        return X_pca

    def getDataSets(self):
        if self.data_flg == 0:
            data_sets = datasets.load_iris()
        elif self.data_flg == 1:
            data_sets = datasets.load_digits()

        self.all_data_sets = data_sets
        self.nrow, self.ncol = data_sets.data.shape
        return data_sets

pca = PCA_CLASS()

