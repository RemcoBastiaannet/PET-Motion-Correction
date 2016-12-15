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

def MLEMrecon(originalImageP, measurementP, nMLEM, forwardprojector, backprojector): 
    # Initial guess 
    guessP = np.ones(np.shape(originalImageP)) # Dit moet waarschijnlijk niet het eerste plaatje zijn. 
    guessS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

    for i in range(nMLEM): 
        # update current guess 
        fillStirSpace(guessS, guessP)

        # Forward project initial guess 
        guessSinogram = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
        forwardprojector.forward_project(guessSinogram, guessS); 
        guessSinogramS = guessSinogram.get_segment_by_sinogram(0)
        guessSinogramP = stirextra.to_numpy(guessSinogramS)

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
        normalizationSinogramS = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
        normalizationSinogram = normalizationSinogramS.get_segment_by_sinogram(0)
        fillStirSpace(normalizationSinogram, normalizationSinogramP) 
        normalizationSinogramS.set_segment(normalizationSinogram)

        normalizationS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

        backprojector.back_project(normalizationS, normalizationSinogramS)
        normalizationP = stirextra.to_numpy(normalizationS)

        # Update guess 
        guessP = stirextra.to_numpy(guessS)
        errorBackprP = stirextra.to_numpy(errorBackprS)
        guessP *= errorBackprP/normalizationP

    return guessP 

nVoxelsXY = 256
nRings = 1
nLOR = 10
nFrames = 5
nMLEM = 5 

# Setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)
span = 1 # No axial compression  
max_ring_diff = 0 # maximum ring difference between the rings of oblique LORs 
trueShiftPixels = 10; 

# Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)

# Phantoms for each time frame
nFrames = 2 
phantomP = [] 

# Create the individual time frames, the phantom is shifted in each frame w.r.t. the previous one 
plt.figure(1)
for iFrame in range(nFrames): 
    tmp = np.zeros((1, 128, 128)) 
    tmp[0, (10+iFrame*trueShiftPixels):(30+iFrame*trueShiftPixels), 60:80] = 1
    phantomP.append(tmp) 
    plt.subplot(1,2,iFrame+1), plt.title('Time frame {0}'.format(iFrame + 1)), plt.xlabel('x'), plt.ylabel('y'), plt.imshow(phantomP[iFrame][0,:,:])
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

plt.figure(2) 
plt.subplot(1,2,1), plt.title('Forward projection, time frame {0}'.format(iFrame + 1)), plt.xlabel('theta'), plt.ylabel('x'), plt.imshow(phantomP[iFrame][0,:,:])
    
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
        # Motion correction, by changing the backprojector such that it will correct for the shift of the forward projector 
        MotionModel.setOffset(nPixelShift) 
        backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)
        MotionModelShiftedImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
        MotionModelShiftedImageP = MLEMrecon(originalImageP, sinogramShiftedGuessP, nMLEM, forwardprojector, backprojector)
        plt.figure(3)
        plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(originalImageP[0,:,:])  
        plt.subplot(1,2,2), plt.title('Motion Model Shifted Second Time Frame'), plt.imshow(MotionModelShiftedImageP[0,:,:]) 
        plt.show() 

        print 'Shifted sinogram was successfully matched to the measurement :)'
        print 'Shift: {0}'.format(nPixelShift), 'Quadratic error: {0}'.format(quadError)
        raw_input("Press Enter to continue...")
        break; 

    nPixelShift = nPixelShift - 1 

    # If no solution is found after a certain number of iterations the loop will be ended 
    if nPixelShift > trueShiftPixels + 5: 
        print 'Shifted sinogram was NOT successfully matched to the measurement... :('
        print nPixelShift
        raw_input("Press Enter to continue...")
        break; 