import sys
import pylab
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skimage.transform import iradon, radon

# Original image
originalImage = np.zeros((128, 128)) # matrix 128 x 128 gevuld met 0'en
for i in range(128): 
    for j in range(128): 
        if (i-40)*(i-40) + (j-40)*(j-40) + 10 < 30: 
            originalImage[i, j] = 1 

plt.figure(), plt.title('Original image'), plt.imshow(originalImage[:,:]), plt.show()

# Forward projection (measurement)
iAngles = np.linspace(0, 360, 120)[:-1]
measurement = radon(originalImage[:,:], iAngles)
plt.figure(), plt.title('Measurement'), plt.imshow(measurement), plt.show()

# Initial guess 
guess = np.ones(np.shape(originalImage))
plt.figure(), plt.title('Initial guess MLEM'), plt.imshow(guess[:,:]), plt.show()

# MLEM loop 
for i in range(nIt): 
    # Forward project initial guess 
    guessSinogram = radon(guess, iAngles) 
    plt.figure(), plt.title('Sinogram of current guess'), plt.imshow(guessSinogram[:,:]), plt.show()    

    # Compare guess to measurement 
    error = measurement/guessSinogram
    error[np.isnan(error)] = 0
    error[np.isinf(error)] = 0
    error[error > 1E10] = 0;
    error[error < 1E-10] = 0

    # Error terugprojecteren 
    errorBackprS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

    backprojector.back_project(errorBackprS, guessSinogram)

    # Normalization - werkt nog niet correct! 
    normalizationSinogramP = np.ones(np.shape(measurementP)) 
    #tmp1 = np.zeros((np.shape(measurementP)[0],np.shape(measurementP)[1]/3,np.shape(measurementP)[2]))
    #tmp2 = np.ones((np.shape(measurementP)[0],np.shape(measurementP)[1]/3,np.shape(measurementP)[2]))
    #tmp3 = np.zeros((np.shape(measurementP)[0],np.shape(measurementP)[1]/3,np.shape(measurementP)[2]))
    #normalizationSinogramP = np.concatenate((tmp1, tmp2, tmp3), axis = 1)
    normalizationSinogramS = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    normalizationSinogram = normalizationSinogramS.get_segment_by_sinogram(0)
    fillStirSpace(normalizationSinogram, normalizationSinogramP) 
    normalizationSinogramS.set_segment(normalizationSinogram)

    normalizationS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

    backprojector.back_project(normalizationS, normalizationSinogramS)

    normalizationP = stirextra.to_numpy(normalizationS)
    #if i == 0: plt.figure(6), plt.title('MLEM normalization'), plt.imshow(normalizationP[0,:,:]), plt.show()

    diagonalProfile = normalizationP[0,:,:].diagonal()
    #if i == 0: plt.figure(7), plt.title('MLEM normalization diagonal'), plt.plot(diagonalProfile), plt.show()
    #print diagonalProfile

    # Update guess 
    guessP = stirextra.to_numpy(guessS)
    errorBackprP = stirextra.to_numpy(errorBackprS)
    guessP *= errorBackprP/normalizationP

    countIt = i+1 # counts the number of iterations (for nIt iterations, i = 0, ..., nIt-1)
    plt.figure(8), plt.title('Guess after {0} iteration(s)'.format(i+1)), plt.imshow(guessP[0,:,:]), plt.show()