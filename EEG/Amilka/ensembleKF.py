# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:12:55 2020

@author: Loreto
"""
import numpy as np
import math
from scipy import linalg as lin
import givensMethod as givens
import time as time 
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
from scipy.signal import welch

start = time.time()

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
            
            SnewT = givens.givensMethod(F, Q, L)
            
        except np.linalg.LinAlgError:
            SnewT = 0
            return SnewT
    else:
        try:
            SnewT = givens.givensMethod(F, Q, P)
        except np.linalg.LinAlgError:
            SnewT = 0
            return SnewT
        
    return SnewT.T #return as S

########################## POTTER ##########################
def potterAlg(S, H, R, x, y):
    lenX = len(x)
    lenH = len(H)
    I = np.eye(lenX)
    S_i = S
    x_i = x
    
    for i in range(lenH):        
        H_i = H if H.shape == (1,lenH) else reshape(H[i], (1,lenX))
        y_i = scalar(y[i]) if y.shape == (lenH,1) else y
        R_i = variance(R[i])
        phi = dotProduct(S_i.T, H_i.T)
        a = scalar(1 / ((dotProduct(phi.T, phi)) + R_i))    
        gammaPlus = a / (1 + math.sqrt(dotProduct(a, R_i)))
        Splus = dotProduct(S_i, (I - (dotProduct(dotProduct(a, gammaPlus), 
                                                 (dotProduct(phi, phi.T))))))
        k = dotProduct(S_i, phi)
        xPlus = x_i + k.dot(a*(y_i-H_i.dot(x_i)))
        
        S_i = Splus
        x_i = xPlus
   
    return S_i,x_i

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
           
           pk, x_preAll = potterAlg(S_t,H,R,xpt,y)
           pk_WC, x_preWC = potterAlg(S_tWC,H_WC,R_WC,xpt_WC,yWC)
           pk_NWC, x_preNWC = potterAlg(S_tNWC,H_NWC,R_NWC,xpt_NWC,yNWC)
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

scalar = lambda x: x.item() if x.size == 1 else x
zeros = np.zeros  
ones = np.ones 
variance = np.var
dotProduct = np.dot
reshape = np.reshape

samplingRate = 128 #SAMPLING RATE

#RELEVANT SENSORS, ACCORDING TO THE POSITION PRESENTED IN SENSORNAME
aW = np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,1]) 

name = 'User1' #NAME OF THE USER

time_log_path = os.path.join(script_dir, f"time_per_session_{name}.txt")
with open(time_log_path, "a") as log_file:
    log_file.write("Session\tTimeTaken(s)\n")  # header

for m in range(2,21): #SESSIONS TO ANALYZE
    session_start = time.time()
    numb = m
    signalName1 = os.path.join(script_dir, name + "_Pre" + str(m) + ".csv")
    signalName2 = os.path.join(script_dir, name + "_Post" + str(m) + ".csv")

    if not os.path.exists(signalName1) or not os.path.exists(signalName2):
        print(f"Skipping session {m}: One or both files are missing.")
        continue  # Skip to the next iteration if files are missing
    
    allPre, originalPre, wcPre, nwcPre, yAll_Pre, yWCPre, yNWCPre = EnKF(signalName1,samplingRate, aW)
    allPost, originalPost, wcPost, nwcPost, yAll_Post, yWCPost, yNWCPost = EnKF(signalName2,samplingRate, aW)
    
    session_end = time.time()
    session_duration = session_end - session_start

    print("DONE " + str(m))

    with open(time_log_path, "a") as log_file:
        log_file.write(f"{m}\tGivensxPotter\t{session_duration:.2f}\n")
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

time_axis_pre = np.arange(len(amplitudePre_All)) / samplingRate
time_axis_ypre = np.arange(len(amplitudeYPre_All)) / samplingRate

# Plot
plt.figure(figsize=(12, 6))
plt.plot(time_axis_pre, amplitudePre_All, label='amplitudePre_All', alpha=0.7)
plt.plot(time_axis_ypre, amplitudeYPre_All, label='amplitudeYPre_All', alpha=0.7)

plt.title("Amplitude vs Time (Pre vs yPre)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Welch PSD for all sessions
plt.figure(figsize=(14, 8))

for session_number in range(2, 21):
    # File names
    pre_original = f'amplitudePre_Originalv002{name}{session_number}.csv'
    ypre_filename = f'yPre_Allv002{name}{session_number}.csv'
    ypreWC_filename = f'yPre_WCv002{name}{session_number}.csv'
    ypreNWC_filename = f'yPre_NWCv002{name}{session_number}.csv'

    post_original = f'amplitudePost_Originalv002{name}{session_number}.csv'
    ypost_filename = f'yPost_Allv002{name}{session_number}.csv'
    ypostWC_filename = f'yPost_WCv002{name}{session_number}.csv'
    ypostNWC_filename = f'yPost_NWCv002{name}{session_number}.csv'

    # Skip session if any file is missing
    required_files = [
        pre_original, ypre_filename, ypreWC_filename, ypreNWC_filename,
        post_original, ypost_filename, ypostWC_filename, ypostNWC_filename
    ]
    if not all(os.path.exists(f) for f in required_files):
        print(f"Skipping session {session_number}: One or more files missing.")
        continue

    # Load data
    amplitudeYPre_All = np.genfromtxt(ypre_filename, delimiter=",")
    amplitudeYPre_NWC = np.genfromtxt(ypreNWC_filename, delimiter=",")
    amplitudeYPre_WC = np.genfromtxt(ypreWC_filename, delimiter=",")
    amplitudeOriginal = np.genfromtxt(pre_original, delimiter=",")

    amplitudeYPost_All = np.genfromtxt(ypost_filename, delimiter=",")
    amplitudeYPost_NWC = np.genfromtxt(ypostNWC_filename, delimiter=",")
    amplitudeYPost_WC = np.genfromtxt(ypostWC_filename, delimiter=",")
    amplitudePostOriginal = np.genfromtxt(post_original, delimiter=",")

    # Compute Welch PSD
    freq_all, psd_all = welch(amplitudeYPre_All, fs=samplingRate, nperseg=512)
    freq_wc, psd_wc = welch(amplitudeYPre_WC, fs=samplingRate, nperseg=512)
    freq_nwc, psd_nwc = welch(amplitudeYPre_NWC, fs=samplingRate, nperseg=512)
    freq_orig, psd_orig = welch(amplitudeOriginal, fs=samplingRate, nperseg=512)

    freq_all_post, psd_all_post = welch(amplitudeYPost_All, fs=samplingRate, nperseg=512)
    freq_wc_post, psd_wc_post = welch(amplitudeYPost_WC, fs=samplingRate, nperseg=512)
    freq_nwc_post, psd_nwc_post = welch(amplitudeYPost_NWC, fs=samplingRate, nperseg=512)
    freq_orig_post, psd_orig_post = welch(amplitudePostOriginal, fs=samplingRate, nperseg=512)

    # Convert to dB
    psd_all_db = 10 * np.log10(psd_all)
    psd_wc_db = 10 * np.log10(psd_wc)
    psd_nwc_db = 10 * np.log10(psd_nwc)
    psd_orig_db = 10 * np.log10(psd_orig)

    psd_all_db_post = 10 * np.log10(psd_all_post)
    psd_wc_db_post = 10 * np.log10(psd_wc_post)
    psd_nwc_db_post = 10 * np.log10(psd_nwc_post)
    psd_orig_db_post = 10 * np.log10(psd_orig_post)

    # Plot side-by-side Pre and Post
    plt.figure(figsize=(14, 6))
    plt.suptitle(f"Session {session_number} - Welch PSD Comparison (Pre vs Post)", fontsize=14)

    # Pre plot
    plt.subplot(1, 2, 1)
    plt.plot(freq_all, psd_all_db, label='yPre_All', linewidth=1.2)
    plt.plot(freq_wc, psd_wc_db, label='yPre_WC', linestyle='--')
    plt.plot(freq_nwc, psd_nwc_db, label='yPre_NWC', linestyle='-.')
    plt.plot(freq_orig, psd_orig_db, label='Original', linestyle=':')
    plt.title("Pre Session 2 Givens x Potter")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.grid(True)
    plt.legend()

    # Post plot
    plt.subplot(1, 2, 2)
    plt.plot(freq_all_post, psd_all_db_post, label='yPost_All', linewidth=1.2)
    plt.plot(freq_wc_post, psd_wc_db_post, label='yPost_WC', linestyle='--')
    plt.plot(freq_nwc_post, psd_nwc_db_post, label='yPost_NWC', linestyle='-.')
    plt.plot(freq_orig_post, psd_orig_db_post, label='Original', linestyle=':')
    plt.title("Post Session 2 Givens x Potter")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True)
    plt.legend()    

    output_filename = (f"GivensPotter_Session{session_number}_PreAndPost.png")
    output_path = os.path.join(script_dir, output_filename)
    plt.savefig(output_path)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()