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
nFrames = 3
nMLEM = 1


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
#if (showImages): plt.figure(1)
for iFrame in range(nFrames): 
    tmp = np.zeros((1, 128, 128)) 
    tmp[0, (10+iFrame*trueShiftPixels):(30+iFrame*trueShiftPixels), 60:80] = 1
    phantomP.append(tmp) 
    #if (showImages): plt.subplot(1,nFrames,iFrame+1), plt.title('Time frame {0}'.format(iFrame + 1)), plt.imshow(phantomP[iFrame][0,:,:])
#if (showImages): plt.show() 

originalImageP = phantomP[0]
originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(originalImageS, originalImageP)

### STIR versies van de verschillende time frames, nodig voor projecties 
phantomS = []
for iFrame in range(nFrames): 
    imageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
    fillStirSpace(imageS, phantomP[iFrame])
    phantomS.append(imageS)
### 

# Initialize the projection matrix (using ray-tracing) 
slope = 0.0 
offSet = 0.0 # Do not shift the first projection (reference frame)  
MotionModel = stir.MotionModel(nFrames, slope, offSet) # A motion model is compulsory  
projmatrix = stir.ProjMatrixByBinUsingRayTracing(MotionModel)
projmatrix.set_num_tangential_LORs(nLOR)
projmatrix.set_up(projdata_info, originalImageS)

####
recon = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config.par')
s = recon.set_up(target)
####

# Create projectors
forwardprojector    = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)

### Measurement/projections
measurementPhantomPlist = []
for iFrame in range(nFrames): 
    measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    forwardprojector.forward_project(measurement, phantomS[iFrame]);  
    measurementS = measurement.get_segment_by_sinogram(0)
    measurementP = stirextra.to_numpy(measurementS)
    measurementPhantomPlist.append(measurementP) 

#plt.figure(2)
#for iFrame in range(nFrames): plt.subplot(2,nFrames/2+1,iFrame+1), plt.imshow(measurementPhantomPlist[iFrame][0,:,:])
#plt.show()
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

# TO DO 
# Juiste shift vinden, in plaats van er in zetten
# Normalisatie fixen


''' 
# 2) Maar hier een projectie van met Offset 0 (frame 1, referentie) 
# 3) Maak hier nog een projectie van, maar gok de Offset, deze is in het algemeen niet 0 (frame 2, verschoven) 
# 4) Combineer de sinogrammen tot een soort time resolved 3D sinogram 
# 5) Vergelijk de opgetelde sinogrammen met je meting 
# 6) Update de gok voor de shift 
# 7) Herhaal totdat de shift gevonden is 
# 8) Motion correctie van de sinogrammen, gecorrigeerde sinogrammen bij elkaar optellen en terugprojecteren (NORMALISATIE)  

for i in range(nMLEM):

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
'''


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