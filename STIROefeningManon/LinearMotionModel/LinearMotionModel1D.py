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

nVoxelsXY = 256
nRings = 1
nLOR = 10
nFrames = 5
nMLEM = 2 

# Setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)
span = 1 # No axial compression  
max_ring_diff = 0 # maximum ring difference between the rings of oblique LORs 
trueShiftPixels = 50; 

# Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)

# Phantoms for each time frame
nFrames = 5 
phantomP = [] 

# Create the individual time frames, the phantom is shifted in each frame w.r.t. the previous one 
plt.figure(1)
for iFrame in range(nFrames): 
    tmp = np.zeros((1, 128, 128)) 
    tmp[0, (10+iFrame*trueShiftPixels):(30+iFrame*trueShiftPixels), 60:80] = 1
    phantomP.append(tmp) 
    plt.subplot(3,3,iFrame+1), plt.title('Original image Time frame {0}'.format(iFrame)), plt.imshow(phantomP[iFrame][0,:,:])
plt.show() 

originalImageP = phantomP[0]
originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(originalImageS, originalImageP)

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

# Forward project the original image 
measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
forwardprojector.forward_project(measurement, originalImageS);  
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)

# Convert Python data to STIR data for the second time frame (= first shifted frame, after the original image) 
measurementShiftedImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
            stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
            stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(measurementShiftedImageS, phantomP[1])

# Forward project the data of the second time frame (= the first shifted frame, after the original image)    
measurementShiftedImage = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
forwardprojector.forward_project(measurementShiftedImage, measurementShiftedImageS);  
    
measurementShiftedImageS = measurementShiftedImage.get_segment_by_sinogram(0)
measurementShiftedImageP = stirextra.to_numpy(measurementShiftedImageS)
    
# Finding the shift of the second frame (first shifted frame) w.r.t. the first frame (original image) and correction for it, using a do-while like loop 
nPixelShift = -1 # Attempted shift 
shiftedImageGuessP = phantomP[0] # First guess for the shifted image

while True: 
    # Update the motion model with a new shift for this iteration 
    MotionModel.setOffset(nPixelShift) 

    # Create STIR space and fill with our first guess, that will be shifted in the forward projection 
    shiftedImageGuessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
    fillStirSpace(shiftedImageGuessS, shiftedImageGuessP) 

    # Forward project (and thus shift) first guess 
    sinogramShiftedGuess = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    forwardprojector.forward_project(sinogramShiftedGuess, shiftedImageGuessS)
    sinogramShiftedGuessS = sinogramShiftedGuess.get_segment_by_sinogram(0)
    sinogramShiftedGuessP = stirextra.to_numpy(sinogramShiftedGuessS) 

    # Comparing the sinograms of the shifted measurement with the shifted guess
    differenceError = measurementShiftedImageP - sinogramShiftedGuessP
    quadError = np.sum(differenceError**2)
    maxError = 500   
      
    if (quadError < maxError): 
        print 'Shifted sinogram was matched to the measurement, with:'
        print 'Shift: {0}'.format(nPixelShift), 'Quadratic error: {0}'.format(quadError)
        
        # Motion correction, by changing the backprojector such that it will correct for the shift of the forward projector 
        MotionModel.setOffset(nPixelShift) 
        backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)
        MotionModelShiftedImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
        backprojector.back_project(MotionModelShiftedImageS, sinogramShiftedGuess)
        MotionModelShiftedImageP = stirextra.to_numpy(MotionModelShiftedImageS)
        plt.figure(30)
        plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(originalImageP[0,:,:])  
        plt.subplot(1,2,2), plt.title('Motion Model Shifted Second Time Frame'), plt.imshow(MotionModelShiftedImageP[0,:,:]) 
        plt.show() 

        raw_input("Press Enter to continue...")
        break; 

    nPixelShift = nPixelShift - 1 

    # If no solution is found after a certain number of iterations the loop will be ended 
    if nPixelShift > trueShiftPixels + 5: 
        print 'Shifted sinogram was NOT successfully matched to the measurement'
        print nPixelShift
        raw_input("Press Enter to continue...")
        break; 






'''
# Backprojecting the sinogram to get an image 
finalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

backprojector.back_project(finalImageS, measurement) 
finalImageP = stirextra.to_numpy(finalImageS)

#plt.figure(3), plt.title('Backprojection original image'), plt.imshow(finalImageP[0,:,:]), plt.show()

# MLEM reconstruction

# Initial guess 
guessP = np.ones(np.shape(originalImageP))
guessS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

#plt.figure(4), plt.title('Initial guess MLEM'), plt.imshow(guessP[0,:,:]), plt.show()

for i in range(nMLEM): 
    # update current guess 
    fillStirSpace(guessS, guessP)

    # Forward project initial guess 
    guessSinogram = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    forwardprojector.forward_project(guessSinogram, guessS); 
    guessSinogramS = guessSinogram.get_segment_by_sinogram(0)
    guessSinogramP = stirextra.to_numpy(guessSinogramS)

    #plt.figure(5), plt.title('Sinogram of current guess'), plt.imshow(guessSinogramP[0,:,:]), plt.show()    

    # Compare guess to measurement 
    errorP = measurementP/guessSinogramP
    errorP[np.isnan(errorP)] = 0
    errorP[np.isinf(errorP)] = 0
    errorP[errorP > 1E10] = 0;
    errorP[errorP < 1E-10] = 0;

    fillStirSpace(guessSinogramS, errorP)
    guessSinogram.set_segment(guessSinogramS)  

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
    if i == 0: plt.figure(6), plt.title('MLEM normalization'), plt.imshow(normalizationP[0,:,:]), plt.show()

    diagonalProfile = normalizationP[0,:,:].diagonal()
    if i == 0: plt.figure(7), plt.title('MLEM normalization diagonal'), plt.plot(diagonalProfile), plt.show()
    print diagonalProfile

    # Update guess 
    guessP = stirextra.to_numpy(guessS)
    errorBackprP = stirextra.to_numpy(errorBackprS)
    guessP *= errorBackprP/normalizationP

    countIt = i+1 # counts the number of iterations (for nMLEM iterations, i = 0, ..., nIt-1)
    #plt.figure(8), plt.title('Guess after {0} iteration(s)'.format(i+1)), plt.imshow(guessP[0,:,:]), plt.show()
'''