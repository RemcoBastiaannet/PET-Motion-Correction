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
from prompt_toolkit import input

showImages = True   

nVoxelsXY = 256
nRings = 1
nLOR = 10
nFrames = 3
nMLEM = 3

# Setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)
span = 1 # No axial compression  
max_ring_diff = 0 # maximum ring difference between the rings of oblique LORs 
trueShiftPixels = 10; # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd)  

# Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)

# Phantoms for each time frame
phantomP = [] 

# Create the individual time frames, the phantom is shifted in each frame w.r.t. the previous one 
for iFrame in range(nFrames): 
    tmp = np.zeros((1, 128, 128)) 
    tmp[0, (10+iFrame*trueShiftPixels):(30+iFrame*trueShiftPixels), 60:80] = 1
    phantomP.append(tmp) 

originalImageP = phantomP[0]
originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(originalImageS, originalImageP)

plt.figure(1)
plt.subplot(1,2,1), plt.title('Phantom frame 1'), plt.imshow(phantomP[0][0,:,:]) 
plt.subplot(1,2,2), plt.title('Phantom frame 2'), plt.imshow(phantomP[1][0,:,:]) 
plt.show()

phantomS = []
for iFrame in range(nFrames): 
    imageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
    fillStirSpace(imageS, phantomP[iFrame])
    phantomS.append(imageS)

# Initialize the projection matrix (using ray-tracing) 
slope = 0.0 
offSet = 0.0 # Do not shift the first projection (reference frame)  
MotionModel = stir.MotionModel(nFrames, slope, offSet) # A motion model is compulsory  
projmatrix = stir.ProjMatrixByBinUsingRayTracing(MotionModel)
projmatrix.set_num_tangential_LORs(nLOR)
projmatrix.set_up(projdata_info, originalImageS)

# Create projectors
forwardprojector    = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)


reconImagePList = []
#_________________________FIRST RECONSTRUCTION________________________
# Measurement/projections of inital time frame
measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
forwardprojector.forward_project(measurement, phantomS[0])
measurement.write_to_file('sino_1.hs')
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)
plt.imshow(measurementP[0,:,:]), plt.title('Sinogram time frame 1'), plt.show()

# Image reconstruction using OSMAPOSL 
reconImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
reconImageS.fill(1)

reconOSMAPOSL = stir.OSMAPOSLReconstruction3DFloat('config_1.par')
reconOSMAPOSL.reconstruct(reconImageS)
reconImageP = stirextra.to_numpy(reconImage)
reconImagePList.append(reconImageP)
plt.imshow(reconImageP[0,:,:]), plt.title('OSMAPOSL Reconstruction time frame 1'), plt.show()


#_________________________SECOND RECONSTRUCTION________________________
# Measurement/projections of inital time frame
measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
forwardprojector.forward_project(measurement, phantomS[1])
measurement.write_to_file('sino_2.hs')
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)
plt.imshow(measurementP[0,:,:]), plt.title('Sinogram time frame 2'), plt.show()

# Image reconstruction using OSMAPOSL 
reconImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
reconImageS.fill(1)

reconOSMAPOSL = stir.OSMAPOSLReconstruction3DFloat('config_2.par')
reconOSMAPOSL.reconstruct(reconImageS)
reconImageP = stirextra.to_numpy(reconImage)
reconImagePList.append(reconImageP)
plt.imshow(reconImageP[0,:,:]), plt.title('OSMAPOSL Reconstruction time frame 2'), plt.show()


#_________________________COMBINING RECONSTRUCTIONS________________________




'''

# Create projectors
forwardprojector    = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)

### Measurement/projections of the inital time frames 
measurementPhantomPlist = []
for iFrame in range(nFrames): 
    measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    forwardprojector.forward_project(measurement, phantomS[iFrame]);  
    measurementS = measurement.get_segment_by_sinogram(0)
    measurementP = stirextra.to_numpy(measurementS)
    measurementPhantomPlist.append(measurementP) 
### 

# MLEM 
# Initial guess 
guessImageP = np.ones(np.shape(originalImageP)) # Dit moet waarschijnlijk niet het eerste plaatje zijn. 
guessImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
guessImageSlist = []
guessImagePlist = []
guessImagePlist.append(guessImageP) 
guessSinogramPlist = []
errorPTotal = 1 

# TEST 
fillStirSpace(guessImageS, guessImageP)
normSinogram = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
forwardprojector.forward_project(normSinogram, guessImageS)
tmp = normSinogram.get_segment_by_sinogram(0)
normSinogramP = stirextra.to_numpy(tmp)
plt.imshow(normSinogramP[0,:,:]), plt.show()
fillStirSpace(tmp, normSinogramP)
normSinogram.set_segment(tmp)
 
backprojector.back_project(guessImageS, normSinogram)

b = stirextra.to_numpy(guessImageS)
plt.imshow(b[0,:,:]), plt.show()
# EINDE TEST 

for i in range(nMLEM): 
    par1 = -10 # Dit is de juiste shift 

    # update current guess 
    fillStirSpace(guessImageS, guessImageP)
    guessImageSlist.append(guessImageS)
 
    # Forward project initial guess  
    for iFrame in range(nFrames): 
        guessSinogram = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
        MotionModel.setOffset(par1*iFrame) # Ieder frame heeft zijn eigen shift (als er meer tijd is verstreken, is de shift groter), hoe groot precies hangt af van een parameter
        forwardprojector.forward_project(guessSinogram, guessImageS)
        guessSinogramS = guessSinogram.get_segment_by_sinogram(0)
        guessSinogramP = stirextra.to_numpy(guessSinogramS)
        guessSinogramPlist.append(guessSinogramP)
        errorP = measurementPhantomPlist[iFrame]/guessSinogramP
        errorP[np.isnan(errorP)] = 0
        errorP[np.isinf(errorP)] = 0
        errorP[errorP > 1E10] = 0
        errorP[errorP < 1E-10] = 0
        plt.figure(3), plt.imshow(errorP[0,:,:]), plt.title('Sinogram error'), plt.show() 

        fillStirSpace(guessSinogramS, errorP)
        guessSinogram.set_segment(guessSinogramS)

        errorBackprS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                        stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                        stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

        MotionModel.setOffset(par1*iFrame)
        backprojector.back_project(errorBackprS, guessSinogram)
        errorBackprP = stirextra.to_numpy(errorBackprS) 
        plt.figure(4), plt.imshow(errorBackprP[0,:,:]), plt.title('Backprojection sinogram error'), plt.show() 
        errorPTotal *= errorBackprP
    #plt.figure(5), plt.imshow(errorPTotal[0,:,:]), plt.title('Total error'), plt.show() 
        #plt.figure(3), plt.subplot(1,3,iFrame+1), plt.imshow(guessSinogramP[0,:,:])
    #plt.show() 

    # Normalization - werkt nog niet correct! 
    normalizationSinogramP = np.ones(np.shape(measurementP)) 
    normalizationSinogramS = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    normalizationSinogram = normalizationSinogramS.get_segment_by_sinogram(0)
    fillStirSpace(normalizationSinogram, normalizationSinogramP) 
    normalizationSinogramS.set_segment(normalizationSinogram)

    normalizationS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

    MotionModel.setOffset(par1*iFrame) # normalisatie moet frame specifiek worden 
    backprojector.back_project(normalizationS, normalizationSinogramS)
    normalizationP = stirextra.to_numpy(normalizationS)
    plt.figure(5), plt.imshow(normalizationP[0,:,:]), plt.title('Normalization'), plt.show() 

    # Update guess 
    guessImageP = stirextra.to_numpy(guessImageS)
    errorBackprP = stirextra.to_numpy(errorBackprS)
    norm = 1/(3*b)
    norm[np.isnan(norm)] = 0
    norm[np.isinf(norm)] = 0
    plt.figure(3), plt.imshow(norm[0,:,:]), plt.show()
    guessImageP *= errorPTotal
    guessImagePlist.append(guessImageP) # voor visualisatie, guessImageP heeft nu de laatste, mocht je die nodig hebben
 
if (showImages): 
    plt.figure(5)
    plt.imshow(guessImagePlist[1][0,:,:])
    plt.show()
'''