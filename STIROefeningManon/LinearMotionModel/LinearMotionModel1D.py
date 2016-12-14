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
nIt = 2 # number of MLEM iterations 

# Now we setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)
span = 1 # axial compression 
max_ring_diff = 0 # maximum ring difference between the rings of oblique LORs 
trueShiftPixels = 10; # Shift per time frame, handig als dit deelbaar is door 5 (want de oplossing wordt gezocht in 5 stappen) 

# Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)

# Phantom for each time frame, in python dataformat  
phantomP = [] 

# Create the individual frames 
plt.figure(1)
for iFrame in range(nFrames): 
    tmp = np.zeros((1, 128, 128)) # matrix 128 x 128 gevuld met 0'en
    tmp[0, (10+iFrame*trueShiftPixels):(30+iFrame*trueShiftPixels), 60:80] = 1
    phantomP.append(tmp) 
    plt.subplot(3,3,iFrame+1), plt.title('Original image Time frame {0}'.format(iFrame)), plt.imshow(phantomP[iFrame][0,:,:])
plt.show() 

originalImageP = phantomP[0]

# Stir data format instance with the size of the original image in python (not yet filled!) 
originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  

# Filling the stir data format with the original image 
fillStirSpace(originalImageS, originalImageP)

# Initialize the projection matrix (using ray-tracing) 
slope = 0.0 # kan eigenlijk weg, want dit zijn de default waardes 
offSet = 0.0 # eerste meting niet verschuiven 
MotionModel = stir.MotionModel(nFrames, slope, offSet) 
projmatrix = stir.ProjMatrixByBinUsingRayTracing(MotionModel)
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

# ALS JE ALLE TIJDSFRAMES WIL HEBBEN DOE JE range(1, nFrames)!! 
nPixelShiftTotal = [] # Total shift for each time rime (w.r.t. the reference time frame) 
nPixelShiftTotal.insert(1, 0) # De eerste keer moet er al iets in zitten, om op te tellen bij de eerste waarde. 
for iFrame in range(1, 4): # de eerste is je referentie, dus je begint met beweging zoeken in het tweede frame (index 1) 
    # "Measuring" the sinogram of the phantom in the current time frame (the time frame we want to find!) 
    measurementShiftedImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
    fillStirSpace(measurementShiftedImageS, phantomP[iFrame])
    
    measurementShiftedImage = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    forwardprojector.forward_project(measurementShiftedImage, measurementShiftedImageS);  
    
    measurementShiftedImageS = measurementShiftedImage.get_segment_by_sinogram(0)
    measurementShiftedImageP = stirextra.to_numpy(measurementShiftedImageS)
    
    nPixelShift = 0 # Het is mogelijk dat er geen shift is in het volgende time frame, dus die situatie moet je ook checken  
    shiftedImageGuessP = phantomP[iFrame-1] # First guess is the previous image 

    while True: # alternative for a do-while loop
        MotionModel.setOffset(nPixelShift)

        shiftedImageGuessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                        stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                        stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
        fillStirSpace(shiftedImageGuessS, shiftedImageGuessP) 

        # Forward project shifted guess 
        sinogramShiftedGuess = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
        forwardprojector.forward_project(sinogramShiftedGuess, shiftedImageGuessS)
        sinogramShiftedGuessS = sinogramShiftedGuess.get_segment_by_sinogram(0)
        sinogramShiftedGuessP = stirextra.to_numpy(sinogramShiftedGuessS) 

        # Comparing 
        differenceError = measurementShiftedImageP - sinogramShiftedGuessP
        quadError = np.sum(differenceError**2)
        maxError = 3000 # ???    
      
        if (quadError < maxError): 
            print 'Shifted sinogram was matched to the measurement, with:'
            print 'shift: {0}'.format(nPixelShift), 'Quadratic error: {0}'.format(quadError)
            raw_input("Press Enter to continue...")
            # Motion Correction (only if the shift is found succesfully) 
            nPixelShiftTotal.insert(len(nPixelShiftTotal)+1, nPixelShift + nPixelShiftTotal[-1]) # KLOPT NIET!!!!!!!!!!!!!!!!!! Want nPixelShift is niet het totaal. 
            # !!!!!!!!!!!!!!!!!!!!!!!!!
            # !!!!!!!!!!!!!!!!!

            MotionModel.setOffset(nPixelShiftTotal[iFrame-1]) # NIET -nPixelShift (dat minteken zorgt ie zelf al voor blijkbaar), iFrame-1 want element 0 hoort bij het eerste verschoven frame t.o.v. het reference frame.
            MotionModelShiftedImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
            backprojector.back_project(MotionModelShiftedImageS, sinogramShiftedGuess) 
            MotionModelShiftedImageP = stirextra.to_numpy(MotionModelShiftedImageS) 
            plt.figure(30), 
            plt.subplot(1,2,1), plt.title('Original Image (0th time frame)'), plt.imshow(phantomP[0][0,:,:]) 
            plt.subplot(1,2,2), plt.title('Backprojection of Motion Model Shifted Image'), plt.imshow(MotionModelShiftedImageP[0,:,:]) 
            plt.show() 
            break; 
        nPixelShift = nPixelShift - trueShiftPixels/5 

        if abs(nPixelShift) > abs(trueShiftPixels*iFrame + 5): 
            print 'Shifted sinogram was NOT successfully matched to the measurement'
            print 'shift: {0}'.format(nPixelShift) 
            raw_input("Press Enter to continue...")
            break; 

    





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

for i in range(nIt): 
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

    countIt = i+1 # counts the number of iterations (for nIt iterations, i = 0, ..., nIt-1)
    #plt.figure(8), plt.title('Guess after {0} iteration(s)'.format(i+1)), plt.imshow(guessP[0,:,:]), plt.show()