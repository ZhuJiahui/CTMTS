# -*- coding: utf-8 -*-
'''
Created on 2014年7月27日

@author: ZhuJiahui506
'''

import os
import time
import numpy as np
from sklearn.cluster import KMeans
from KLD import SKLD

def get_max_eig(L, m, withev=False):
    '''
    
    :param L:
    :param m:
    '''
    eigen_values, eigen_vectors = np.linalg.eig(L)
    
    # 降序排序
    idx = eigen_values.argsort()
    idx = idx[::-1]
    
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[idx] 
    
    max_eigen_values = eigen_values[0 : m]
    max_eigen_vectors = eigen_vectors[0 : m, :]
    
    #正向化
    max_eigen_vectors = np.abs(max_eigen_vectors)
    
    if withev:
        return max_eigen_values, max_eigen_vectors
    else:
        return max_eigen_vectors

def spectral_cluster(data, cluster_number):
    dimension = len(data)
    withev = False
    
    W1 = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(i, dimension):
            W1[i, j] = 1.0 / (SKLD(data[i], data[j]) + 1.0)
            W1[j, i] = W1[i, j]
    
    D1 = np.zeros((dimension, dimension))
   
    for i in range(dimension):
        D1[i, i] = 1.0 / np.sqrt(np.sum(W1[i]))
  
    L1 = np.dot(np.dot(D1, W1), D1)
 
    U1 = get_max_eig(L1, cluster_number, withev)
    
    U1 = U1.transpose()
    k_means = KMeans(init='k-means++', n_clusters=cluster_number, n_init=10)
    k_means.fit(U1)
    k_means_labels = k_means.labels_
    
    return k_means_labels

def spectral_cluster2(W1, cluster_number):
    dimension = len(W1)
    withev = False
    
    D1 = np.zeros((dimension, dimension))
   
    for i in range(dimension):
        D1[i, i] = 1.0 / np.sqrt(np.sum(W1[i]))
  
    L1 = np.dot(np.dot(D1, W1), D1)
 
    U1 = get_max_eig(L1, cluster_number, withev)
    
    U1 = U1.transpose()
    k_means = KMeans(init='k-means++', n_clusters=cluster_number, n_init=10)
    k_means.fit(U1)
    k_means_labels = k_means.labels_
    
    return k_means_labels
    

if __name__ == '__main__':
    pass