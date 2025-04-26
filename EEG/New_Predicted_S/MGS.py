# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 20:13:36 2021

@author: Loreto
"""

import numpy as np
import math

#A1 = np.array([[1, 0], [1, 1], [0, -1], [0, -1]])
#A1 = np.array([[2, 1], [2, 3], [1, 1], [0, np.sqrt(2)]])

def mgsMethod(F,Q,S):

    Ftrans = F.T
    Strans = S.T
    Qtrans = (np.sqrt(Q)).T
    
    mul = Ftrans.dot(Strans)
    
    At = np.concatenate((mul, Qtrans), axis = 0)
    
    n = np.size(At[0])
    W = np.zeros([n,n])
    Apt = np.zeros([len(At),len(At[1])])
    
    for k in range (0,n):
        Akk = At[:,k]
        sigma = np.sqrt(np.dot(Akk.T, Akk))
        
        for j in range(0,n):
            if(j == k):
                W[k][j] = sigma
                continue
            if(j in range(0, k)):
                W[k][j] = 0
                continue
            if(j in range(k+1, n)):
                W[k][j] = np.dot(Akk.T, At[:,j])/sigma
                continue
            
        for j in range(k+1, n):
            Apt[:,j] = At[:,j] - ((np.dot(W[k][j], Akk)) /sigma)
            
        At = Apt
    
    return W

#ss = mgsMethod(A1)