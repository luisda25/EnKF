# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:12:55 2020

@author: Loreto
"""
import numpy as np
import math
from scipy import linalg as lin


#New_predicted_S
import New_Predicted_S.Givens_Rotation_Liz as givens
import New_Predicted_S.MGS as mgs
import New_Predicted_S.gram as gs
#State_transition_matrix
import State_Trans_Matrix.Potter_Liz as potter
import State_Trans_Matrix.bierman as bierman
import State_Trans_Matrix.biermanLiz as biermanLiz

#import Jocelyn
import Codigo_Jocelyn.bierman_hh as bierman_hh

########################## READ THE SIGNAL ##########################
def readSignal(nameSignal, samplingRate):
    my_data = np.genfromtxt(nameSignal, delimiter=',')
    x = np.delete(my_data, (0), axis=0)
    sessionMatrix = []
    
    i = 0
    j = samplingRate
    
    for s in range(math.floor(len(x)/samplingRate)):
        sessionMatrix.append(np.asarray(x[i:j]))
        i = i + samplingRate
        j = j + samplingRate
        
    return np.transpose(sessionMatrix, (0,2,1))

########################## CALCULATE F (TAYLOR) ##########################
def calculateF_Taylor(samplingRate, numSensors):
    c = np.zeros([numSensors,numSensors])
       
    for i in range(numSensors):
        v = (1/samplingRate**i)/math.factorial(i)
        a = np.empty(numSensors)
        a.fill(v)
        np.fill_diagonal(c[i:], a)
        
    return c.T

########################## GET R MATRIX ##########################
def noiseDiagCov(noise):
    noiseC = zeros([len(noise), len(noise)])
    np.fill_diagonal(noiseC, [np.cov(num) for num in noise])
    return noiseC

####################### INITIAL COVARIANCE MATRIX ########################
def initialCovMatrix(wholeSignal):
    return np.cov(wholeSignal)

####################### GET H MATRIX ########################
def getH(wC, totalSensors):
    result = np.asarray(np. where(wC == 1))
    matrix = np.zeros([len(result[0]),totalSensors])
    for i in range(0, len(result[0])):
        matrix[i][result[0][i]] = 1
    return matrix

################### CONCATENATE AMPLITUDE ###########################

def concatenateAmplitude(listSignal):
    if len(listSignal) == 0:
        return zeros([len(sensorName)])
    temp=listSignal[0]
    for i in range(1, len(listSignal)):
        temp = np.concatenate((temp, listSignal[i]), axis=None)
    return temp

########################## SQUARE ROOT MATRIX ##########################
def getNextSquareRoot(P,Q,F, typeM):
    SnewT = 0
    newP = P
    if (typeM == 'initial'):
        if(np.allclose(P, P.T) == 'false'):     #KNOW IF A MATRIX IS SYMMETRIC
            newP = (P + P.T)/2
        
        try:
            lu, d, perm = lin.ldl(newP, lower = 1)
            L = lu.dot(lin.fractional_matrix_power(d,0.5))# P = SS.T = LD^(1/2) D^(1/2) L.T
            
            SnewT = gs.modified_gram_schmidt(F, Q, L)
            
        except np.linalg.LinAlgError:
            SnewT = 0
            return SnewT
    else:
        try:
            SnewT = gs.modified_gram_schmidt(F, Q, P)
        except np.linalg.LinAlgError:
            SnewT = 0
            return SnewT
        
    return SnewT.T #return as S

#################### ENKF ################################
def EnKF(name_Signal, samplingRate, wC):
    numberSensors = len(wC)
    invertWC = np.where(wC == 1, 0, 1) # VECTOR FOR NOT CONSIDERING THE WINNER COMBINATION
    
    signal = readSignal(name_Signal, samplingRate) #READ OF THE CSV FILE
    
    resultAll = zeros([len(signal), samplingRate])
    resultOriginal = zeros([len(signal), samplingRate])
    resultWC = zeros([len(signal), samplingRate])
    resultNWC = zeros([len(signal), samplingRate])
    
    resultAll_Temp = zeros([1,samplingRate])
    resultOriginal_Temp = zeros([1,samplingRate])
    resultWC_Temp = zeros([1,samplingRate])
    resultNWC_Temp = zeros([1,samplingRate])
      
    H = np.eye(numberSensors) #CONSIDER ALL SENSORS
    H_WC = getH(wC, numberSensors) #CONSIDER RELEVANT SENSORS
    H_NWC = getH(invertWC, numberSensors) #NO CONSIDER RELEVANT SENSORS
  
    Q = np.eye(numberSensors)
    wNoise = zeros((numberSensors,1)) 
    
    counterN = 1
    lastN = samplingRate-1
    
    F = calculateF_Taylor(samplingRate, numberSensors)
    F_WC = calculateF_Taylor(samplingRate, numberSensors)
    np.fill_diagonal(F_WC[0:], wC)
    
    F_NWC = calculateF_Taylor(samplingRate, numberSensors)
    np.fill_diagonal(F_NWC[0:], invertWC)
    
    matrixState = signal[0]
    
    initialP = initialCovMatrix(matrixState)
    pk = initialP
    pk_WC = initialP
    pk_NWC = initialP
    ty = 'initial'
     
    x_preAll = np.zeros([numberSensors,1])
    x_preWC = np.zeros([numberSensors,1])
    x_preNWC = np.zeros([numberSensors,1])
    
    yResult = []
    yResult_WC =[]
    yResult_NWC =[]
    
    
    for i in range(len(signal)):
        
        matrixState = signal[i]
  
        for j in range(samplingRate):
 
           x = x_preAll
           xWC = x_preWC
           xNWC = x_preNWC
           
           zNoiseMatrix = np.random.normal(0, 1, size = (numberSensors, numberSensors)) 
           zNoiseMatrixWC = np.random.normal(0, 1, size = (3, 3)) 
           zNoiseMatrixNWC = np.random.normal(0, 1, size = (numberSensors-3, numberSensors-3)) 
    
           R = noiseDiagCov(zNoiseMatrix)
           R_WC = noiseDiagCov(zNoiseMatrixWC)
           R_NWC = noiseDiagCov(zNoiseMatrixNWC)
           
           
           xpt = (dotProduct(F, x)) + wNoise
           xpt_WC = (dotProduct(F_WC, xWC)) + wNoise
           xpt_NWC = (dotProduct(F_NWC, xNWC)) + wNoise
           
           resultAll_Temp[0,j] = scalar(np.sum(xpt))/numberSensors 
           resultWC_Temp[0,j] = scalar(np.sum(xpt_WC))/3 
           resultNWC_Temp[0,j] = scalar(np.sum(xpt_NWC))/(numberSensors-3) 
           
           S_t = getNextSquareRoot(pk, Q, F,ty)
           S_tWC = getNextSquareRoot(pk_WC, Q, F_WC,ty)
           S_tNWC = getNextSquareRoot(pk_NWC, Q, F_NWC,ty)
           
           if(j != lastN):
               nextState = matrixState[:,[counterN]]
               counterN+=1
                
           if(j == samplingRate-1 and i != len(signal)-1):
                tempM = signal[i+1]
                nextState = tempM[:,[0]]
                
           if(j == samplingRate-1 and i == len(signal)-1):
                tempM = signal[0]
                nextState = tempM[:,[0]]
                
           resultOriginal_Temp[0,j] = np.sum(nextState)/numberSensors
        
           y = nextState 
           yWC = H_WC.dot(nextState)
           yNWC = H_NWC.dot(nextState)
           
           yResult.append(np.sum(nextState)/numberSensors)
           yResult_WC.append(np.sum(yWC)/3)
           yResult_NWC.append(np.sum(yNWC)/(numberSensors-3))
           
           pk, x_preAll = potter.Potter_Algorithm(S_t,H,R,xpt,y)
           pk_WC, x_preWC = potter.Potter_Algorithm(S_tWC,H_WC,R_WC,xpt_WC,yWC)
           pk_NWC, x_preNWC = potter.Potter_Algorithm(S_tNWC,H_NWC,R_NWC,xpt_NWC,yNWC)
           ty = 'other'        
        
        resultAll[i] = resultAll_Temp
        resultOriginal[i] = resultOriginal_Temp
        resultWC[i] = resultWC_Temp
        resultNWC[i] = resultNWC_Temp
        
        
        resultAll_Temp = zeros([1,samplingRate])    
        resultOriginal_Temp = zeros([1,samplingRate])
        resultWC_Temp = zeros([1,samplingRate])
        resultNWC_Temp = zeros([1,samplingRate])
        counterN = 1
        print(i)
        
    return resultAll, resultOriginal, resultWC, resultNWC, yResult,yResult_WC, yResult_NWC
           
#####################################################################################################
#NAME OF THE SENSORS USED. THE EXAMPLE IS RELATED TO EMOTIV EPOC+.
sensorName = np.array(['AF3','F7','F3','FC5','T7','P7','O1',
                       'O2','P8','T8','FC6','F4','F8','AF4'])

scalar = lambda x: x.item()
zeros = np.zeros  
ones = np.ones 
variance = np.var
dotProduct = np.dot
reshape = np.reshape

samplingRate = 128 #SAMPLING RATE

#RELEVANT SENSORS, ACCORDING TO THE POSITION PRESENTED IN SENSORNAME
aW = np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,1]) 

name = 'Anakaren' #NAME OF THE USER

for m in range(2,21): #SESSIONS TO ANALYZE
    numb = m
    signalName1 = name + "_Pre" + str(m) + ".csv"
    signalName2 = name + "_Post" + str(m) + ".csv"
    
    allPre, originalPre, wcPre, nwcPre, yAll_Pre, yWCPre, yNWCPre = EnKF(signalName1,samplingRate, aW)
    allPost, originalPost, wcPost, nwcPost, yAll_Post, yWCPost, yNWCPost = EnKF(signalName2,samplingRate, aW)
    
    print("DONE " + str(m))
##################################################################################
    amplitudePre_All = concatenateAmplitude(allPre)
    amplitudePre_Original = concatenateAmplitude(originalPre)
    amplitudePre_WC = concatenateAmplitude(wcPre)
    amplitudePre_NWC = concatenateAmplitude(nwcPre)
    
    amplitudePost_All = concatenateAmplitude(allPost)
    amplitudePost_Original = concatenateAmplitude(originalPost)
    amplitudePost_WC = concatenateAmplitude(wcPost)
    amplitudePost_NWC = concatenateAmplitude(nwcPost)
    
##################################################################################
    
    amplitudeYPre_All = np.array(yAll_Pre)
    amplitudeYPre_WC = np.array(yWCPre)
    amplitudeYPre_NWC = np.array(yNWCPre)
    
    amplitudeYPost_All = np.array(yAll_Post)
    amplitudeYPost_WC = np.array(yWCPost)
    amplitudeYPost_NWC = np.array(yNWCPost)
    
##################################################################################    
    
    np.savetxt('amplitudePre_Allv002' + name + str(numb) + '.csv', amplitudePre_All, delimiter=",")
    np.savetxt('amplitudePre_Originalv002'+ name + str(numb) + '.csv', amplitudePre_Original, delimiter=",")
    np.savetxt('amplitudePre_WCv002'+ name + str(numb) + '.csv', amplitudePre_WC, delimiter=",")
    np.savetxt('amplitudePre_NWCv002'+ name + str(numb) + '.csv', amplitudePre_NWC, delimiter=",")
    
    np.savetxt('amplitudePost_Allv002' + name + str(numb) + '.csv', amplitudePost_All, delimiter=",")
    np.savetxt('amplitudePost_Originalv002'+ name + str(numb) + '.csv', amplitudePost_Original, delimiter=",")
    np.savetxt('amplitudePost_WCv002'+ name + str(numb) + '.csv', amplitudePost_WC, delimiter=",")
    np.savetxt('amplitudePost_NWCv002'+ name + str(numb) + '.csv', amplitudePost_NWC, delimiter=",")
    
 ##################################################################################      
    
    np.savetxt('yPre_Allv002' + name + str(numb) + '.csv', amplitudeYPre_All, delimiter=",")
    np.savetxt('yPre_WCv002'+ name + str(numb) + '.csv', amplitudeYPre_WC, delimiter=",")
    np.savetxt('yPre_NWCv002'+ name + str(numb) + '.csv', amplitudeYPre_NWC, delimiter=",")
    
    np.savetxt('yPost_Allv002' + name + str(numb) + '.csv', amplitudeYPost_All, delimiter=",")
    np.savetxt('yPost_WCv002'+ name + str(numb) + '.csv', amplitudeYPost_WC, delimiter=",")
    np.savetxt('yPost_NWCv002'+ name + str(numb) + '.csv', amplitudeYPost_NWC, delimiter=",")
    