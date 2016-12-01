# TO DO 
# STIR-master -> examples -> Matlab code, Python, mMR
# - Sinogram 
# - Reconstructie met 1) scatter correctie, 2) randoms correctie, 3) attenuatie correctie 
# - OSMAPOSL proberen werkend te krijgen

import sys
import stir
import stirextra
import pylab
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from StirSupport import *
from scipy.optimize import minimize

nVoxelsXY = 256
nRings = 1
nLOR = 10
nFrames = 15
nIt = 3 # number of MLEM iterations 

#Now we setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)
span = 1 
max_ring_diff = 0 # maximum ring difference between the rings of oblique LORs 

# Span is a number used by CTI to say how much axial compression has been used. 
# It is always an odd number. Higher span, more axial compression. Span 1 means no axial compression.
# In 3D PET, an axial compression is used to reduce the data size and the computation times during the image reconstruction. 
# This is achieved by averaging a set of sinograms with adjacent values of the oblique polar angle. This sampling scheme achieves good results in the centre of the FOV. 
# However, there is a loss in the radial, tangential and axial resolutions at off-centre positions, which is increased in scanners with large FOVs. 

# CTI: www.cti-medical.co.uk ?

#Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)

# Original image python dataformat  
originalImageP = np.zeros((1, 128, 128)) # matrix 128 x 128 gevuld met 0'en
for i in range(128): 
    for j in range(128): 
        if (i-40)*(i-40) + (j-40)*(j-40) + 10 < 30: 
            originalImageP[0, i, j] = 1 

# Stir data format instance with the size of the original image in python (not yet filled!) 
originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

# Filling the stir data format with the original image 
fillStirSpace(originalImageS, originalImageP)

# Initialize the projection matrix (using ray-tracing)
projmatrix = stir.ProjMatrixByBinUsingRayTracing()
projmatrix.set_num_tangential_LORs(nLOR)
projmatrix.set_up(projdata_info, originalImageS)

# Create projectors
forwardprojector    = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)

# Creating an instance for the sinogram (measurement), it is not yet filled 
measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)

# Forward project originalImageS and store in measurement 
forwardprojector.forward_project(measurement, originalImageS);  

# Converting the stir sinogram to a numpy sinogram 
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)

#plt.figure(1)
#plt.imshow(measurementP[0,:,:], cmap = plt.cm.Greys_r, interpolation = None, vmin = 0)
#plt.show() # program pauses until the figure is closed!

# Backprojecting the sinogram to get an image 
finalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

backprojector.back_project(finalImageS, measurement) 
finalImageP = stirextra.to_numpy(finalImageS)

#plt.figure(2)
#plt.imshow(finalImageP[0,:,:], cmap = plt.cm.Greys_r, interpolation = None, vmin = 0)
#plt.show() # program pauses until the figure is closed!



# MLEM reconstructie - poing 2 (werkt ook nog niet) 
# guess *= backproject[ measured projection/(forward projection of current estimate) ] * normalization 

# Initial guess 
guessP = np.ones(np.shape(originalImageP))

for i in range(nIt): 
    guessS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
    fillStirSpace(guessS, guessP)

    #plt.figure(3)
    #plt.imshow(guessP[0,:,:], cmap = plt.cm.Greys_r, interpolation = None, vmin = 0)
    #plt.show() # program pauses until the figure is closed!

    # Forward project initial guess 
    guess_sinogram = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    forwardprojector.forward_project(guess_sinogram, guessS); 
    guess_sinogramS = guess_sinogram.get_segment_by_sinogram(0)
    guess_sinogramP = stirextra.to_numpy(guess_sinogramS)

    #plt.figure(4)
    #plt.imshow(guess_sinogramP[0,:,:], cmap = plt.cm.Greys_r, interpolation = None, vmin = 0)
    #plt.show() # program pauses until the figure is closed!

    # Measured projection is gewoon measurementS (of measurementP) 

    # Compare guess to measurement 
    errorP = measurementP/guess_sinogramP
    errorP[np.isnan(errorP)] = 0
    errorP[np.isinf(errorP)] = 0
    errorP[errorP > 1E10] = 0;
    errorP[errorP < 1E-10] = 0;

    # error moet uiteindelijk ProjData zijn  
    # of: er wel eerst een discretised density van maken, vullen met fillStirSpace en daarna terug naar projdata
    fillStirSpace(guess_sinogramS, errorP)
    guess_sinogram.set_segment(guess_sinogramS)  

    # Error terugprojecteren 
    errorBackprS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

    backprojector.back_project(errorBackprS, guess_sinogram)

    # Normalization 
    # Vul sinogram met 0 
    # Extraheer hier een numpyarray uit (voor de grootte) 
    # Maak dezelfde numpyarray, maar vul deze met 1'en 
    # Gebruik fillStirSpace (die vult alleen binnen de grenzen van het sinogram, met sinogram.fill(1) zou je iets anders krijgen) 
    
    #sinogram_tmpS = guess_sinogramS 
    #sinogram_tmpP = stirextra.to_numpy(guess_sinogramS) 
    #sinogram_tmpP2 = np.ones(np.shape(sinogram_tmpP)) 
    #normalizationS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    #stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    #stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(guessP)[0],np.shape(guessP)[1],np.shape(guessP)[2] ))) 
    #backprojector.back_project(normalizationS, sinogram_tmpP2) 
    #fillStirSpace(normalizationS, sinogram_tmpP2) # hier gaat het fout! 

    # Update guess 
    guessP = stirextra.to_numpy(guessS)
    errorBackprP = stirextra.to_numpy(errorBackprS)
    guessP *= errorBackprP

    plt.figure(5)
    plt.imshow(guessP[0,:,:], cmap = plt.cm.Greys_r, interpolation = None, vmin = 0)
    plt.show() # program pauses until the figure is closed!