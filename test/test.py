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
guessS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
fillStirSpace(guessS, guessP)

plt.figure(3)
plt.imshow(guessP[0,:,:], cmap = plt.cm.Greys_r, interpolation = None, vmin = 0)
plt.show() # program pauses until the figure is closed!

# Forward project initial guess 
guess_sinogram = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
forwardprojector.forward_project(guess_sinogram, guessS); 
guess_sinogramS = guess_sinogram.get_segment_by_sinogram(0)
guess_sinogramP = stirextra.to_numpy(guess_sinogramS)
# Ik heb het idee dat het sinogram niet klopt... 

plt.figure(4)
plt.imshow(guess_sinogramP[0,:,:], cmap = plt.cm.Greys_r, interpolation = None, vmin = 0)
plt.show() # program pauses until the figure is closed!


# Measured projection is gewoon measurementS (of measurementP) 

# Compare guess to measurement 
error = measurementP/guess_sinogramP
# Dit vind ie niet leuk 

# Normalization 
# sinogram vult met 0, numpyarray uit extraheren (voor de size), zelfde numpyarray maken die bestaat uit alleen maar 1'en. 
# Dan pak je stir fill volume ding van Remco. Die vult alleen binnen de grenzen vna het sinogram (dus anders dan wanneer je sinogram.fill(1) zou doen) 

# Update guess 




##############################################################################################################################################################

## MLEM reconstructie - poging 1 (werkt niet) 
## guess *= backproject[ measured projection/(forward projection of current estimate) ] * normalization 

## Initial guess 
#guess      = stir.floatvoxelsoncartesiangrid(projdata_info, 1,
#                stir.floatcartesiancoordinate3d(stir.make_floatcoordinate(0,0,0)),
#                stir.intcartesiancoordinate3d(stir.make_intcoordinate(np.shape(originalimagep)[0],np.shape(originalimagep)[1],np.shape(originalimagep)[2] ))) 
#guess.fill(1) 

## Forward project initial guess 
#guessSinogram = stir.projdatainmemory(stir.examinfo(), projdata_info)
#forwardprojector.forward_project(guessSinogram, guess)

#sinocomp = guessSinogram.get_segment_by_sinogram(0)

## Measurement 
#meascomp = measurement.get_segment_by_sinogram(0)

#reconspace  = stir.floatvoxelsoncartesiangrid(projdata_info, 1,
#                stir.floatcartesiancoordinate3d(stir.make_floatcoordinate(0,0,0)),
#                stir.intcartesiancoordinate3d(stir.make_intcoordinate(np.shape(originalimagep)[0],np.shape(originalimagep)[1],np.shape(originalimagep)[2] ))) 

#forwardprojector.forward_project(measurement, reconspace)

## Compare initial guess to measurement (calculate error) 
#error = meascomp/sinocomp
#error[np.isnan(error)] = 0
#error[np.isinf(error)] = 0
#error[error > 1e10] = 0;
#error[error < 1e-10] = 0;

#errors = stir.floatvoxelsoncartesiangrid(projdata_info, 1,
#                stir.floatcartesiancoordinate3d(stir.make_floatcoordinate(0,0,0)),
#                stir.intcartesiancoordinate3d(stir.make_intcoordinate(np.shape(originalimagep)[0],np.shape(originalimagep)[1],np.shape(originalimagep)[2] )))

## Normalization
##sinogram vult met 0, numpyarray uit extraheren (voor de size), zelfde numpyarray maken die bestaat uit alleen maar 1'en. 
##Dan pak je stir fill volume ding van Remco. Die vult alleen binnen de grenzen vna het sinogram (dus anders dan wanneer je sinogram.fill(1) zou doen) 

## Update guess using the error 

#guess *= backprojector.back_project(fillstirspace(errors, error))