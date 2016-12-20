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
    # originalImageP is only used for the shape of the geuss 
    # measurementP is used to compare the guess with 

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

   
        # Niet zo handig/overzichtelijk, maar in guessSinogram zit nu dus de error 
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

showImages = True   

nVoxelsXY = 256
nRings = 1
nLOR = 10
nFrames = 2
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
nFrames = 2 
phantomP = [] 

# Create the individual time frames, the phantom is shifted in each frame w.r.t. the previous one 
if (showImages): plt.figure(1)
for iFrame in range(nFrames): 
    tmp = np.zeros((1, 128, 128)) 
    tmp[0, (10+iFrame*trueShiftPixels):(30+iFrame*trueShiftPixels), 60:80] = 1
    phantomP.append(tmp) 
    if (showImages): plt.subplot(1,2,iFrame+1), plt.title('Time frame {0}'.format(iFrame + 1)), plt.xlabel('x'), plt.ylabel('y'), plt.imshow(phantomP[iFrame][0,:,:])
if (showImages): plt.show() 

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

if (showImages): 
    plt.figure(2) 
    plt.subplot(1,2,1), plt.title('Time Frame 1'), plt.imshow(originalImageP[0,:,:])
    plt.subplot(1,2,2), plt.title('Time Frame {0}'.format(iFrame + 1)), plt.xlabel('theta'), plt.ylabel('x'), plt.imshow(measurementShiftedImageP[0,:,:])
    plt.show()

# ######################################### NIEUWE MODEL OPTIMALISATIE (MET MLEM) ########################################
    
# MLEM + model optimalisatie tegelijk 
# 1) Gok je plaatje (begin met alles 1) 
# 2) Maar hier een projectie van met Offset 0 (frame 1, referentie) 
# 3) Maak hier nog een projectie van, maar gok de Offset, deze is in het algemeen niet 0 (frame 2, verschoven) 
# 4) Tel de sinogrammen bij elkaar op (NORMALISATIE) 
# 5) Vergelijk de opgetelde sinogrammen met je meting 
# 6) De error projecteer je terug om je gok te updaten (MLEM), en update je gok voor de shift (eerst nog even niet afhankelijk van hoe de error er precies uitziet) 
# 7) Herhaal totdat de shift gevonden is 
# 8) Motion correctie van de sinogrammen, gecorrigeerde sinogrammen bij elkaar optellen en terugprojecteren (NORMALISATIE) 
# 9) Uitbreiden voor meerdere tijdsframes 

# 1) Gok je plaatje (begin met alles 1) 
imageGuessP = np.ones(np.shape(originalImageP)) # Dit moet waarschijnlijk niet het eerste plaatje zijn. 
imageGuessS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

for i in range(nMLEM):
    fillStirSpace(imageGuessS, imageGuessP)

    # 2) Maar hier een projectie van met Offset 0 (frame 1, referentie) 
    # 3) Maak hier nog een projectie van, maar gok de Offset, deze is in het algemeen niet 0 (frame 2, verschoven) 
    sinogramsGuessS = [] 
    sinogramsGuessP = []

    for iOffset in range(nFrames): 
        MotionModel.setOffset(iOffset) # width of a time bin is now always 1 
        sinogramImageGuess = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
        forwardprojector.forward_project(sinogramImageGuess, imageGuessS)
        sinogramImageGuessS = sinogramImageGuess.get_segment_by_sinogram(0)
        sinogramsGuessS.append(sinogramImageGuessS)
        sinogramsGuessP.append(stirextra.to_numpy(sinogramImageGuessS)) 

    if (showImages):
        plt.figure(3)
        for i in range(nFrames): plt.subplot(1,nFrames,i+1), plt.title('Guess Time Frame {0}'.format(i)), plt.imshow(sinogramsGuessP[i][0,:,:])
        plt.show()  

    # 4) Tel de sinogrammen bij elkaar op (NORMALISATIE) 
    # Deze stap is fout! Je moet de sinogrammen niet bij elkaar optellen, maar in een 3D matrix opslaan, zodat je tijdinformatie bewaart 

    # b) MLEM error 
    errorMLEMP = measurementP/sinogramsGuessP[0]
    errorMLEMP[np.isnan(errorMLEMP)] = 0
    errorMLEMP[np.isinf(errorMLEMP)] = 0
    errorMLEMP[errorMLEMP > 1E10] = 0;
    errorMLEMP[errorMLEMP < 1E-10] = 0;

    # 6) De error projecteer je terug om je gok te updaten (MLEM), en update je gok voor de shift (eerst nog even niet afhankelijk van hoe de error er precies uitziet) 
    errorMLEM = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    errorMLEMS = errorMLEM.get_segment_by_sinogram(0)
    fillStirSpace(errorMLEMS, errorMLEMP)

    errorBackprS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

    MotionModel.setOffset(0.0)
    backprojector.back_project(errorBackprS, errorMLEM)

    normalizationSinogramP = np.ones(np.shape(measurementP)) 
    normalizationSinogramS = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    normalizationSinogram = normalizationSinogramS.get_segment_by_sinogram(0)
    fillStirSpace(normalizationSinogram, normalizationSinogramP) 
    normalizationSinogramS.set_segment(normalizationSinogram)

    normalizationS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

    MotionModel.setOffset(0.0)
    backprojector.back_project(normalizationS, normalizationSinogramS)
    normalizationP = stirextra.to_numpy(normalizationS)

    # Update guess 
    errorBackprP = stirextra.to_numpy(errorBackprS)
    imageGuessP *= errorBackprP/normalizationP

    plt.figure(3)
    plt.title('Guess'), plt.imshow(imageGuessP[0,:,:])
    plt.show()



'''
#################################################### OUDE MODELOPTIMALISATIE #################################################

# Finding the shift of the second frame (first shifted frame) w.r.t. the first frame (original image) and correction for it, using a do-while like loop 
nPixelShift = 0 # Attempted shift 
shiftedImageGuessP = phantomP[0] # First guess for the shifted image
quadErrorList = []
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
    if (showImages):
        plt.figure(3)
        plt.subplot(1,3,1), plt.title('Time Frame 1'), plt.imshow(measurementP[0,:,:])
        plt.subplot(1,3,2), plt.title('Guess Time Frame 2'), plt.imshow(sinogramShiftedGuessP[0,:,:])
        plt.subplot(1,3,3), plt.title('Difference'), plt.imshow(abs(measurementShiftedImageP[0,:,:]-sinogramShiftedGuessP[0,:,:]))
        plt.show()

    # Comparing the sinograms of the shifted measurement with the shifted guess
    differenceError = measurementShiftedImageP - sinogramShiftedGuessP
    quadError = np.sum(differenceError**2)
    quadErrorList.append(quadError)
    maxError = 500   

    # Backprojection 
    MotionModel.setOffset(0.0)
    backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)
    shiftedImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))
    backprojector.back_project(shiftedImageS, sinogramShiftedGuess)
    shiftedImageP = stirextra.to_numpy(shiftedImageS)

    if (showImages):
        plt.figure(4) 
        plt.subplot(1,2,1), plt.title('Time Frame 2'), plt.imshow(phantomP[1][0,:,:])
        plt.subplot(1,2,2), plt.title('Shifted Time Frame 2'), plt.imshow(shiftedImageP[0,:,:])
        plt.show() 
      
    if (quadError < maxError): 
        # Motion correction, by changing the backprojector such that it will correct for the shift of the forward projector 
        MotionModel.setOffset(nPixelShift) 
        correctedImageP = MLEMrecon(originalImageP, sinogramShiftedGuessP, nMLEM, forwardprojector, backprojector)
        if (showImages):
            plt.figure(5)
            plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(originalImageP[0,:,:])  
            plt.subplot(1,2,2), plt.title('Time Frame 2 Motion Corrected'), plt.imshow(correctedImageP[0,:,:]) 
            plt.show() 

        print 'Shifted sinogram was successfully matched to the measurement :)'
        print 'Shift: {0}'.format(nPixelShift), 'Quadratic error: {0}'.format(quadError)
        #print quadErrorList
        raw_input("Press Enter to continue...")
        break; 

    nPixelShift = nPixelShift - trueShiftPixels/5 

    # If no solution is found after a certain number of iterations the loop will be ended 
    if nPixelShift > trueShiftPixels + 5: 
        print 'Shifted sinogram was NOT successfully matched to the measurement... :('
        print nPixelShift
        raw_input("Press Enter to continue...")
        break; 
'''