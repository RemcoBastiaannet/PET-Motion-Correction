import sys
import pylab
import math
import os
import time
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy.optimize import minimize
from skimage.transform import iradon, radon, rescale
from skimage import data_dir
from skimage.io import imread
from prompt_toolkit import input
import ManonsFunctions as mf 

#phantom = 'Block'
phantom = 'Shepp-Logan' 
noise = False
#noise = True
#motion = 'Step' 
motion = 'Sine'

nIt = 10 
trueShiftAmplitude = 30 # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd) + LET OP: kan niet groter zijn dan de lengte van het plaatje (kan de code niet aan) 
trueOffset = 5
numFigures = 0 

if (motion == 'Step'): nFrames = 2
else: nFrames = 4

figSaveDir = mf.make_figSaveDir(motion, phantom, noise)

# Phantom 
image = mf.make_Phantom(phantom)

plt.figure(), plt.title('Original image'), plt.imshow(image, interpolation = None, vmin = 0, vmax = 1)
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close() 
 
phantomList = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, image)[0]
originalImage = phantomList[0]

for i in range(nFrames):    
    plt.subplot(2,nFrames/2+1,i+1), plt.title('Time frame {0}'.format(i)), plt.imshow(phantomList[i][0,:,:], interpolation=None, vmin = 0, vmax = 1) 
plt.suptitle('Phantom')
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close() 

# Sinusoidal motion 
    plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Sinusoidal phantom shifts'), plt.xlabel('Time frame'), plt.ylabel('Shift')
    plt.plot(range(nFrames), shiftList, label = 'True motion')
    plt.legend(loc = 4)
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_shiftList.png'.format(numFigures, trueShiftAmplitude))
    numFigures += 1 
    plt.close()

    for i in range(nFrames):    
        plt.figure(figsize=(5.0, 5.0))
        plt.title('{0}'.format(i)), plt.imshow(phantomList[i][0,:,:], interpolation=None, vmin = 0) 
        plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantomFrame{}.png'.format(numFigures, trueShiftAmplitude, i))
        numFigures += 1 
        plt.close() 
    
    plt.figure(figsize=(23.0, 21.0))
    for i in range(nFrames):    
        plt.subplot(2,nFrames/2+1,i+1), plt.title('{0}'.format(i)), plt.imshow(phantomList[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0, vmax = 1) 
    plt.suptitle('Phantom')
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude))
    numFigures += 1 
    plt.close() 
# End sinusoidal motion 

# Forward projection (measurement)
iAngles = np.linspace(0, 360, 120)[:-1]
measurement = radon(originalImage[0,:,:], iAngles)

# Initial guess 
guess = np.ones(np.shape(originalImage))

# Normalization - werkt nog niet correct! 
normSino = np.ones(np.shape(measurement))
norm = iradon(normSino, iAngles, filter = None) # We willen nu geen ramp filter
plt.figure(), plt.title('MLEM normalization'), plt.imshow(norm, interpolation = None, vmin = 0, vmax = 0.03)
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_norm.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close() 
    
diagonalProfile = norm.diagonal()

# MLEM loop 
for iIt in range(nIt): 
    # Forward project initial guess 
    guessSinogram = radon(guess[0,:,:], iAngles) 

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
plt.figure(), plt.title('Guess after {0} iteration(s)'.format(iIt+1)), plt.imshow(guess[0,:,:], interpolation = None, vmin = 0, vmax = 1)
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_finalImage.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close() 

plt.figure() 
plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(originalImage[0,:,:], interpolation=None, vmin = 0, vmax = 1) 
plt.subplot(1,2,2), plt.title('Reconstructed Image'), plt.imshow(guess[0,:,:], interpolation=None, vmin = 0, vmax = 1) 
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_originalAndRecon.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close() 