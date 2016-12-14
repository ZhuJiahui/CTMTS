# -*- coding: utf-8 -*-
'''
Created on 2014年7月27日

@author: ZhuJiahui506
'''

import numpy as np

    
def sym(X):
    return (X + X.transpose()) / 2.0

def RMSE(X1, X2):
    eX = X1 - X2
    error = 0.0
    for i in range(len(eX)):
        for j in range(len(eX[0])):
            error += (eX[i, j] * eX[i, j])
    
    return np.sqrt(error / len(eX) / len(eX[0]))

def get_IR(W1, W2):
    
    dimension = len(W1)
    
    V1 = W1  #W1的值已改变
    V2 = W2
    
    G1 = np.zeros((dimension, dimension))
    G2 = np.zeros((dimension, dimension))

    for i in range(dimension):
        V1[i, i] = 0.0
        V2[i, i] = 0.0
        G1[i, i] = 1.0 / np.sum(V1[i])
        G2[i, i] = 1.0 / np.sum(V2[i])
    
    #概率转移矩阵
    P1 = np.dot(G1, V1)
    P2 = np.dot(G2, V2)
    
    
    ir_alpha = np.true_divide(dimension, (np.sum(V1) + dimension))
    ir_beta = np.true_divide(dimension, (np.sum(V2) + dimension))
    
    print ir_alpha, ir_beta
    
    A = (1 - ir_alpha) * (1 - ir_beta) * np.dot(P1, P2)
    B = ir_alpha * (1 - ir_beta) * P2 + ir_beta * np.eye(dimension)
    
    IR = np.dot(np.linalg.inv(np.eye(dimension) - A), B)
    IR = sym(IR)
    
    return IR
    

if __name__ == '__main__':
    W1 = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])
    W2 = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.3], [0.1, 0.3, 1.0]])
    
    IRM = get_IR(W1, W2)
    print IRM