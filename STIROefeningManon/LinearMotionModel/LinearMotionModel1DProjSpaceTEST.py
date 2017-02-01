import sys
import stir
import stirextra
import pylab
import numpy as np
import math 
import os
import time
import matplotlib.pyplot as plt
from StirSupport import *
from scipy.optimize import minimize
from prompt_toolkit import input

import scipy as sp
from skimage.io import imread
from skimage import data_dir
from skimage.transform import iradon, radon, rescale

nVoxelsXY = 256 # ?
nRings = 1
nLOR = 10 # ? 
span = 1 # No axial compression  
max_ring_diff = 0 # maximum ring difference between the rings of oblique LORs 
trueShiftPixels = 10; ## # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd) + LET OP: kan niet groter zijn dan de lengte van het plaatje (kan de code niet aan) 

# Setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)

# Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)


#_______________________PHANTOM______________________________________
# Python 
phantomP = [] 

# Shepp-Logan phantom 
imageSmall = imread(data_dir + "/phantom.png", as_grey=True)
imageSmall = rescale(imageSmall, scale=0.4)

tmpY = np.zeros((50, np.shape(imageSmall)[1])) # extend image in the  y-direction, to prevent problems with shifting the image
image = np.concatenate((tmpY, imageSmall), axis = 0)
image = np.concatenate((image, tmpY), axis = 0)

tmpX = np.zeros((np.shape(image)[0], 50))
image = np.concatenate((tmpX, image), axis = 1)
image = np.concatenate((image, tmpX), axis = 1)

# Block phantom 
'''
image = np.zeros((160,160))
image[65:95, 65:95] = 1 
'''


## TEST:  Recons maken na verschillende iteraties -> som berekenen 
testList = []
sumList = [] 
test      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(1,200,200)))  # Lengte moet je gokken/weten 

for i in range(8): 
    test = test.read_from_file('output_config_Proj_1_{0}.hv'.format(i+1))
    testList.append(stirextra.to_numpy(test)) 
    sumList.append(np.sum(stirextra.to_numpy(test)[0,:,:]))

axisX = range(1,9,1)
plt.plot(axisX, sumList, axisX, [np.sum(image)]*len(axisX)), plt.title('Sum of OSMAPOSL recon at different iterations'), plt.xlabel('Iteration number'), plt.show()

for i in range(8): 
    plt.subplot(2,4,i+1), plt.imshow(testList[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Recon {0}'.format(i))
plt.show() 
## 


# Image shape ## 
Nx = np.shape(image)[1] # Als het goed is kloppen x en y zo (maar het is een vierkant plaatje, dus je ziet het niet als het fout gaat...) 
Ny = np.shape(image)[0] 

# Sinusoidal motion 
'''
nFrames = 30
nCycles = 3 # Wordt nu nog even niet gebruikt 
shiftList = [] 
for iFrame in range(nFrames): 
    shift = int(math.sin(nCycles*2*math.pi*iFrame/(nFrames-1))*trueShiftPixels) # nFrames-1 since iFrame never equals nFrame
    shiftList.append(shift) 
    tmp = np.zeros((1, Ny, Nx))
    tmp[0] = image  
    
    if shift > 0: 
        tmp[0, shift:Ny, :] = tmp[0, 0:(Ny-shift), :]
        tmp[0, 0:shift, :] = 0
       
    if shift < 0: 
        tmp[0, 0:(Ny+shift), :] = tmp[0, (-shift):Ny, :] # Be careful with signs as the shift itself is now already negative 
        tmp[0, (Ny+shift):Ny, :] = 0

    phantomP.append(tmp) 
originalImageP = phantomP[0]

plt.plot(shiftList), plt.title('Sinusoidal phantom shifts'), plt.xlabel('Time frame'), plt.ylabel('Shift')
plt.savefig('./Plaatjes/shifts.png')
plt.show()

nFrames = 30 
for i in range(nFrames):    
    plt.figure(figsize=(5.0, 5.0))
    plt.title('{0}'.format(i)), plt.imshow(phantomP[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0) 
    plt.savefig('./Plaatjes/Plaatjes_voor_movieSinusAllFrames/sinusFrame_{0}.png'.format(i))
    
nFrames = 30
plt.figure(figsize=(23.0, 21.0))
for i in range(nFrames):    
    plt.subplot(3,10,i+1), plt.title('{0}'.format(i)), plt.imshow(phantomP[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0) 
plt.savefig('./Plaatjes/sinusAllFrames.png')
'''

# Step function 
nFrames = 2 
for iFrame in range(nFrames): 
    shift = iFrame*trueShiftPixels
    tmp = np.zeros((1, Ny, Nx))
    tmp[0] = image  

    if shift > 0: 
        tmp[0, shift:Ny, :] = tmp[0, 0:(Ny-shift), :]
        tmp[0, 0:shift, :] = 0
       
    if shift < 0: 
        tmp[0, 0:(Ny+shift), :] = tmp[0, (-shift):Ny, :] # Be careful with signs as the shift itself is now already negative 
        tmp[0, (Ny+shift):Ny, :] = 0

    phantomP.append(tmp) 
originalImageP = phantomP[0]

for i in range(nFrames):    
    plt.subplot(1,2,i+1), plt.title('Phantom at time {0}'.format(i)), plt.imshow(phantomP[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0) 
plt.show() 

# STIR 
originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(originalImageS, originalImageP)

phantomS = []

image1S      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(image1S, phantomP[0])
phantomS.append(image1S)

image2S      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(image2S, phantomP[1])
phantomS.append(image2S)


#_______________________PROJ MATRIX AND PROJECTORS______________________
slope = 0.0 
offSet = 0.0 
MotionModel = stir.MotionModel(nFrames, slope, offSet) # A motion model is compulsory  
projmatrix = stir.ProjMatrixByBinUsingRayTracing(MotionModel)
projmatrix.set_num_tangential_LORs(nLOR)
projmatrix.set_up(projdata_info, originalImageS)

# Create projectors
forwardprojector    = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)


#_________________________MEASUREMENT_______________________________
measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
measurementListP = [] 

## First time frame 
MotionModel.setOffset(0.0)
forwardprojector.forward_project(measurement, phantomS[0])
measurement.write_to_file('sinoMeas_1.hs')
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)
measurementListP.append(measurementP) 

## Second time frame 
MotionModel.setOffset(0.0) # Beweging zit al in het plaatje 
forwardprojector.forward_project(measurement, phantomS[1])
measurement.write_to_file('sinoMeas_2.hs')
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)
measurementListP.append(measurementP) 


#_________________________GUESS_______________________________
'''Negeer voor nu het initial estimate'''

##
projection = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)

reconGuess1S = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
reconGuess2S = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

MotionModel.setOffset(0.0)
reconGuess1S.fill(1) # moet er staan
recon1 = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_1.par')
recon1.set_up(reconGuess1S)
recon1.reconstruct(reconGuess1S)

MotionModel.setOffset(0.0)
reconGuess2S.fill(1) # moet er staan
recon2 = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_2.par')
recon2.set_up(reconGuess2S)
recon2.reconstruct(reconGuess2S)

guess1P = stirextra.to_numpy(reconGuess1S)
#plt.imshow(guess1P[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Normal reconstruction, no motion'), plt.show() 
guess2P = stirextra.to_numpy(reconGuess2S)

plt.subplot(1,2,1), plt.imshow(guess1P[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Recon time frame 1')
plt.subplot(1,2,2), plt.imshow(guess2P[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Recon time frame 2')
plt.show()

guessP = 0.5*(guess1P + guess2P)
plt.imshow(guessP[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Initial guess')
plt.savefig('./Plaatjes/Testen_shift_vinden/8_iteraties_{}_true_shift/sigaarInitialGuess.png'.format(trueShiftPixels))
plt.show() 

guessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
fillStirSpace(guessS, guessP)
## 


#_________________________MOTION MODEL OPTIMIZATION_______________________________
quadErrorSumList = []

offSets = range(trueShiftPixels,0,1)

##
offset = -5
## 

## for offset in offSets: 
## FOR 
projectionPList = []

MotionModel.setOffset(+offset) # Is this also the right sign if the real shift is negative? 
forwardprojector.forward_project(projection, guessS)
projection.write_to_file('sino_1.hs')
projectionS = projection.get_segment_by_sinogram(0)
projectionP = stirextra.to_numpy(projectionS)
projectionPList.append(projectionP)

MotionModel.setOffset(-offset) # Is this also the right sign if the real shift is negative? 
forwardprojector.forward_project(projection, guessS)
projection.write_to_file('sino_2.hs')
projectionS = projection.get_segment_by_sinogram(0)
projectionP = stirextra.to_numpy(projectionS)
projectionPList.append(projectionP)

quadErrorSum = np.sum((projectionPList[0][0,:,:] - measurementListP[0][0,:,:])**2) + np.sum((projectionPList[1][0,:,:] - measurementListP[1][0,:,:])**2)
    
quadErrorSumList.append({'offset' : offset, 'quadErrorSum' : quadErrorSum})
## END FOR 

## 
plt.subplot(1,3,1), plt.imshow(projectionPList[0][0,:,:]), plt.title('Guess with + offset'),
plt.subplot(1,3,2), plt.imshow(measurementListP[0][0,:,:]), plt.title('Measurement')
plt.subplot(1,3,3), plt.imshow(abs(measurementListP[0][0,:,:]-projectionPList[0][0,:,:])), plt.title('Difference')
plt.savefig('./Plaatjes/Testen_shift_vinden/8_iteraties_{}_true_shift/sigaarShift{}ProjectionFirstTimeFrame.png'.format(trueShiftPixels, offset))
plt.show()

plt.subplot(1,3,1), plt.imshow(projectionPList[1][0,:,:]), plt.title('Guess with - offset')
plt.subplot(1,3,2), plt.imshow(measurementListP[1][0,:,:]), plt.title('Measurement')
plt.subplot(1,3,3), plt.imshow(abs(measurementListP[1][0,:,:]-projectionPList[1][0,:,:])), plt.title('Difference')
plt.savefig('./Plaatjes/Testen_shift_vinden/8_iteraties_{}_true_shift/sigaarShift{}ProjectionSecondTimeFrame.png'.format(trueShiftPixels, offset))
plt.show() 
## 

print quadErrorSum

'''
quadErrorSums = [x['quadErrorSum'] for x in quadErrorSumList]
for i in range(len(quadErrorSumList)): 
    if(quadErrorSumList[i]['quadErrorSum'] == min(quadErrorSums)): 
        offsetFound = quadErrorSumList[i]['offset']
'''
## plt.plot(offSets, quadErrorSums), plt.title('Quadratic error vs. offset'), plt.show()





'''
#_________________________TEST GUESS PERFECT BLOKJE_______________________________ 

## 
projection = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)

guessP = np.zeros((1, 160,160))
guessP[0, (65+trueShiftPixels/2):(95+trueShiftPixels/2), 65:95] = 1 
plt.imshow(guessP[0,:,:]), plt.title('Initial guess')
plt.savefig('./Plaatjes/Testen_shift_vinden/8_iteraties_{}_true_shift/blokjeInitialGuess.png'.format(trueShiftPixels))
plt.show()

guessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
fillStirSpace(guessS, guessP)
## 
'''

#_________________________TEST GUESS PERFECT SHEPP-LOGAN_______________________________ 
## 
projection = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)

shift = int(0.5*iFrame*trueShiftPixels)
tmp = np.zeros((1, Ny, Nx))
tmp[0] = image  

if shift > 0: 
    tmp[0, shift:Ny, :] = tmp[0, 0:(Ny-shift), :]
    tmp[0, 0:shift, :] = 0
       
if shift < 0: 
    tmp[0, 0:(Ny+shift), :] = tmp[0, (-shift):Ny, :] # Be careful with signs as the shift itself is now already negative 
    tmp[0, (Ny+shift):Ny, :] = 0

guessP = tmp

plt.imshow(guessP[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Initial guess')
plt.savefig('./Plaatjes/Testen_shift_vinden/8_iteraties_{}_true_shift/perfectInitialGuess.png'.format(trueShiftPixels))
plt.show()

guessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
fillStirSpace(guessS, guessP)

## 

#_________________________MOTION MODEL OPTIMIZATION_______________________________
quadErrorSumList = []

offSets = range(trueShiftPixels,0,1)

##
offset = 7
## 

## for offset in offSets: 
## FOR 
projectionPList = []

MotionModel.setOffset(+offset) # Is this also the right sign if the real shift is negative? 
forwardprojector.forward_project(projection, guessS)
projection.write_to_file('sino_1.hs')
projectionS = projection.get_segment_by_sinogram(0)
projectionP = stirextra.to_numpy(projectionS)
projectionPList.append(projectionP)

MotionModel.setOffset(-offset) # Is this also the right sign if the real shift is negative? 
forwardprojector.forward_project(projection, guessS)
projection.write_to_file('sino_2.hs')
projectionS = projection.get_segment_by_sinogram(0)
projectionP = stirextra.to_numpy(projectionS)
projectionPList.append(projectionP)

quadErrorSum = np.sum((projectionPList[0][0,:,:] - measurementListP[0][0,:,:])**2) + np.sum((projectionPList[1][0,:,:] - measurementListP[1][0,:,:])**2)
    
quadErrorSumList.append({'offset' : offset, 'quadErrorSum' : quadErrorSum})
## END FOR 

## 
plt.subplot(1,3,1), plt.imshow(projectionPList[0][0,:,:]), plt.title('Guess with + offset')
plt.subplot(1,3,2), plt.imshow(measurementListP[0][0,:,:]), plt.title('Measurement')
plt.subplot(1,3,3), plt.imshow(abs(measurementListP[0][0,:,:]-projectionPList[0][0,:,:])), plt.title('Difference')
plt.savefig('./Plaatjes/Testen_shift_vinden/8_iteraties_{}_true_shift/perfectGuessShift{}ProjectionFirstTimeFrame.png'.format(trueShiftPixels, offset))
plt.show() 

plt.subplot(1,3,1), plt.imshow(projectionPList[1][0,:,:]), plt.title('Guess with - offset')
plt.subplot(1,3,2), plt.imshow(measurementListP[1][0,:,:]), plt.title('Measurement')
plt.subplot(1,3,3), plt.imshow(abs(measurementListP[1][0,:,:]-projectionPList[1][0,:,:])), plt.title('Difference')
plt.savefig('./Plaatjes/Testen_shift_vinden/8_iteraties_{}_true_shift/perfectGuessShift{}ProjectionSecondTimeFrame.png'.format(trueShiftPixels, offset))
plt.show() 
## 

print quadErrorSum

'''
quadErrorSums = [x['quadErrorSum'] for x in quadErrorSumList]
for i in range(len(quadErrorSumList)): 
    if(quadErrorSumList[i]['quadErrorSum'] == min(quadErrorSums)): 
        offsetFound = quadErrorSumList[i]['offset']
'''
## plt.plot(offSets, quadErrorSums), plt.title('Quadratic error vs. offset'), plt.show()



#_________________________MOTION COMPENSATION_______________________________
reconFrame1S = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  

MotionModel.setOffset(+offsetFound) 
reconFrame1S.fill(1) # moet er staan
recon1 = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_1.par')
recon1.set_up(reconFrame1S)
recon1.reconstruct(reconFrame1S)

reconFrame2S = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

MotionModel.setOffset(-offsetFound) 
reconFrame2S.fill(1) # moet er staan
recon2 = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_2.par')
recon2.set_up(reconFrame2S)
recon2.reconstruct(reconFrame2S)

reconFrame1P = stirextra.to_numpy(reconFrame1S)
reconFrame2P = stirextra.to_numpy(reconFrame2S)

guessP = 0.5*(reconFrame2P[0,:,:]+reconFrame1P[0,:,:])

plt.imshow(guessP[:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Motion corrected reconstruction'), plt.show()