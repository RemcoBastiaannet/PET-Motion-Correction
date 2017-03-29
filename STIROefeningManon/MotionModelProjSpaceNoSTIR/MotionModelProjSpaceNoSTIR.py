import sys
import pylab
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skimage.transform import iradon, radon

nIt = 10

# Original image
originalImage = np.zeros((128, 128)) # matrix 128 x 128 gevuld met 0'en
for i in range(128): 
    for j in range(128): 
        if (i-40)*(i-40) + (j-40)*(j-40) + 10 < 30: 
            originalImage[i, j] = 1 

plt.figure(), plt.title('Original image'), plt.imshow(originalImage, interpolation = None, vmin = 0, vmax = 1), plt.show()

# Forward projection (measurement)
iAngles = np.linspace(0, 360, 120)[:-1]
measurement = radon(originalImage, iAngles)

# Initial guess 
guess = np.ones(np.shape(originalImage))

# Normalization - werkt nog niet correct! 
normSino = np.ones(np.shape(measurement))
norm = iradon(normSino, iAngles, filter = None) # We willen nu geen ramp filter
plt.figure(), plt.title('MLEM normalization'), plt.imshow(norm, interpolation = None, vmin = 0, vmax = 0.03), plt.show()
    
diagonalProfile = norm.diagonal()

# MLEM loop 
for iIt in range(nIt): 
    # Forward project initial guess 
    guessSinogram = radon(guess, iAngles) 

    # Compare guess to measurement 
    error = measurement/guessSinogram
    error[np.isnan(error)] = 0
    error[np.isinf(error)] = 0
    error[error > 1E10] = 0;
    error[error < 1E-10] = 0

    # Error terugprojecteren 
    errorBck = iradon(error, iAngles, filter = None) 

    # Update guess 
    guess *= errorBck/norm
    countIt = iIt+1 # counts the number of iterations
plt.figure(), plt.title('Guess after {0} iteration(s)'.format(iIt+1)), plt.imshow(guess, interpolation = None, vmin = 0, vmax = 1), plt.show()