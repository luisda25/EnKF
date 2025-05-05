# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:31:06 2019

@author: 141614
"""

import numpy as np
import math

# THE GIVENS ROTATION IS USED FOR CALCULATING THE NEXT SQUARE ROOT MATRIX S

# [c -s    [a   = [r
#  s  c] *  b]     0]

# WHERE r = SQRT(a^2 + b^2); c = a/r; s = -b/r 

##########################################################################################

def Givens_Rotation(F,Q,S):
    Ftrans = F.T
    Strans = S.T
    Qtrans = (np.sqrt(Q)).T
    
    mul = Ftrans.dot(Strans)
    
    L = np.concatenate((mul, Qtrans), axis = 0)
    R = np.copy(L)
    
    for j in range(0, L.shape[1]):
        for i in range(L.shape[0],j+1, -1):
            gMatrix = np.identity(L.shape[0])
            a = R[i-2,j]
            b = R[i-1,j]
            if b == 0:
                cos = 1
                sin = 0
            else:
                if abs(b) > abs(a):
                    r = a/b
                    sin = 1/math.sqrt(1 + pow(r,2))
                    cos = sin*r
                else:
                    r = b/a
                    cos = 1/math.sqrt(1 + pow(r,2))
                    sin = cos*r
                    
            gMatrix[i-2:i,i-2:i] = [[cos, -sin], [sin, cos]]
            R = (gMatrix.T).dot(R)
    
    Strans = R[0:len(F)][0:len(F)]
    return Strans