import sys
import pylab
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skimage.transform import iradon, radon

nIt = 3
figSaveDir = './Figures/'

# Original image
originalImage = np.zeros((128, 128)) # matrix 128 x 128 gevuld met 0'en
for i in range(128): 
    for j in range(128): 
        if (i-40)*(i-40) + (j-40)*(j-40) + 10 < 30: 
            originalImage[i, j] = 1 

plt.figure(), plt.title('Original image'), plt.imshow(originalImage[:,:]), plt.show()

'''
for i in range(nFrames):    
    plt.subplot(1,2,i+1), plt.title('Time frame {0}'.format(i)), plt.imshow(phantomP[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0) 
plt.suptitle('Phantom')
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close() 
'''

# Forward projection (measurement)
iAngles = np.linspace(0, 360, 120)[:-1]
measurement = radon(originalImage[:,:], iAngles)
plt.figure(), plt.title('Measurement'), plt.imshow(measurement), plt.show()

# Initial guess 
guess = np.ones(np.shape(originalImage))
plt.figure(), plt.title('Initial guess MLEM'), plt.imshow(guess[:,:]), plt.show()

# MLEM loop 
for iIt in range(nIt): 
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
    errorBck = iradon(error, iAngles) 

    # Normalization - werkt nog niet correct! 
    normSino = np.ones(np.shape(measurement))
    norm = iradon(normSino, iAngles)
    if iIt == 0: plt.figure(), plt.title('MLEM normalization'), plt.imshow(norm), plt.show()
    
    diagonalProfile = norm.diagonal()
    if iIt == 0: plt.figure(), plt.title('MLEM normalization diagonal'), plt.plot(diagonalProfile), plt.show()

    # Update guess 
    guess *= errorBck/norm
    countIt = iIt+1 # counts the number of iterations
    plt.figure(), plt.title('Guess after {0} iteration(s)'.format(iIt+1)), plt.imshow(guess[:,:]), plt.show()