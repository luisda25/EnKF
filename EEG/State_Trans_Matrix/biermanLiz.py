# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:35:33 2021

@author: Loreto
"""

#BIERMAN UD FILTERING
import numpy as np
import math
from scipy import linalg as lin

def biermanFilter(S, H, R, x, y):
    
    originalP = np.dot(S, S.T)
    U, D, perm = lin.ldl(originalP, lower = 0)
    b,p = np.zeros([len(x),1]), np.zeros([len(x),1])
    H_temp = np.reshape(H, (1,len(x))) if H.shape == (len(x),) else np.reshape(H.diagonal(), (1,len(x)))
    
    
    #H_temp = np.reshape(H, (1,len(x))) 
    
    Dnew, Unew = np.zeros([len(x),len(x)]), np.zeros([len(x),len(x)])
    
    f = np.dot(U.T, H_temp.T) #n x 1
    a = R[0,0]
    v = np.asarray([D[i,i]*f[i] for i in range(len(x))])
    
    
    for k in range(len(x)):
        aNew = a + float(f[k] * v[k])
        Dnew[k,k] = D[k,k] * (a/aNew)
        b[k] = v[k]
        p[k] = float(-f[k]/a)
        for j in range(k):
            Unew[j,k] = float(U[j,k] + (b[j] * p[k]))
            b[j] = float(b[j] + (U[j,k] * v[k]))
        a = aNew
    
    K = b/a
    
    y_Pre = np.sum(y)/(np.count_nonzero(H_temp == 1))
    xNew = x + K.dot(y_Pre - H_temp.dot(x))
    
    Snew = (Unew.T.dot(lin.fractional_matrix_power(Dnew,0.5))) #lower
               
    return Snew, xNew