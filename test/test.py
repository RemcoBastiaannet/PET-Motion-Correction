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

#Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)

# Original image python dataformat  
originalImageP = np.zeros((1, 128, 128)) # matrix 128 x 128 gevuld met 0'en
for i in range(128): 
    for j in range(128): 
        if (i-30)*(i-30) + (j-30)*(j-30) < 30: 
            originalImageP[0, i, j] = 1 

# Stir data format instance with the size of the original image in python (not yet filled!) 
originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

# Filling the stir data format with the original image 
fillStirSpace(originalImageS, originalImageP)

# Creating an instance for the sinogram 
sinogram = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)

#Initialize the projection matrix (using ray-tracing)
projmatrix = stir.ProjMatrixByBinUsingRayTracing()
projmatrix.set_num_tangential_LORs(nLOR)
projmatrix.set_up(projdata_info, originalImageS)

#Create projectors
forwardprojector    = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)

# Forward project originalImageS and store in sinogram 
forwardprojector.forward_project(sinogram, originalImageS);  

# Converting the stir sinogram to a numpy sinogram 
sinogramNP = stirextra.to_numpy(sinogram.get_segment_by_sinogram(0))

plt.figure(1)
plt.imshow(sinogramNP[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0)
plt.show()