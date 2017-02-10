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

plt.ioff() # Turn interactive plotting off 

nVoxelsXY = 256
nRings = 1
nLOR = 10 
span = 1 # No axial compression  
max_ring_diff = 0 # maximum ring difference between the rings of oblique LORs 
trueShiftPixels = 30 # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd) + LET OP: kan niet groter zijn dan de lengte van het plaatje (kan de code niet aan) 
numFigures = 0 
nIt = 5 # number of nested EM iterations (model, OSMAPOSL, model, OSMAPOSL, etc.) 

phantom = 'Shepp-Logan' 
#phantom = 'Block'
#noise = True
noise = False
#motion = 'Step' 
motion = 'Sine' 

# Make sure all possible directories exist! 
figSaveDir = './Figures/Sigar/'
if (motion == 'Step'): figSaveDir += 'Step/'
elif (motion == 'Sine'): figSaveDir += 'Sine/'
if (phantom == 'Block'): figSaveDir += 'Block/'
elif (phantom == 'Shepp-Logan'): figSaveDir += 'Shepp-Logan/'
if (noise): figSaveDir += 'Noise/'
else: figSaveDir+= 'No_Noise/'

# Setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)

# Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)


#_______________________PHANTOM______________________________________
# Python 
phantomP = [] 

if (phantom == 'Block'): 
    image = np.zeros((160,160))
    image[65:95, 65:95] = 1 
elif (phantom == 'Shepp-Logan'): 
    imageSmall = imread(data_dir + "/phantom.png", as_grey=True)
    imageSmall = rescale(imageSmall, scale=0.4)

    tmpY = np.zeros((50, np.shape(imageSmall)[1])) # extend image in the  y-direction, to prevent problems with shifting the image
    image = np.concatenate((tmpY, imageSmall), axis = 0)
    image = np.concatenate((image, tmpY), axis = 0)

    tmpX = np.zeros((np.shape(image)[0], 50))
    image = np.concatenate((tmpX, image), axis = 1)
    image = np.concatenate((image, tmpX), axis = 1)

# Image shape 
Nx = np.shape(image)[1] 
Ny = np.shape(image)[0] 

# Step function 
if (motion == 'Step'): 
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
        plt.subplot(1,2,i+1), plt.title('Time frame {0}'.format(i)), plt.imshow(phantomP[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0) 
    plt.suptitle('Phantom')
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftPixels))
    numFigures += 1 
    plt.close() 

# Sinusoidal motion
if (motion == 'Sine'): 
    nFrames = 30
    nCycles = 3
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
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_shiftList.png'.format(numFigures, trueShiftPixels))
    numFigures += 1 
    plt.close()

    for i in range(nFrames):    
        plt.figure(figsize=(5.0, 5.0))
        plt.title('{0}'.format(i)), plt.imshow(phantomP[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0) 
        plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantomFrame{}.png'.format(numFigures, trueShiftPixels, i))
    
    plt.figure(figsize=(23.0, 21.0))
    for i in range(nFrames):    
        plt.subplot(3,10,i+1), plt.title('{0}'.format(i)), plt.imshow(phantomP[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0) 
    plt.suptitle('Phantom')
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftPixels))


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
if (noise): 
    measurementP = sp.random.poisson(measurementP)
measurementListP.append(measurementP) 

## Second time frame 
MotionModel.setOffset(0.0) # Beweging zit al in het plaatje 
forwardprojector.forward_project(measurement, phantomS[1])
measurement.write_to_file('sinoMeas_2.hs')
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)
if (noise): 
    measurementP = sp.random.poisson(measurementP)
measurementListP.append(measurementP) 

plt.subplot(1,2,1), plt.imshow(measurementListP[0][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Meas. TF0')
plt.subplot(1,2,2), plt.imshow(measurementListP[1][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Meas. TF1')
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_Measurements.png'.format(numFigures, trueShiftPixels)) 
numFigures += 1 
plt.close()



#_________________________GUESS_______________________________
'''Negeer voor nu het initial estimate'''
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
guess2P = stirextra.to_numpy(reconGuess2S)
guessP = 0.5*(guess1P + guess2P)
plt.imshow(guessP[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Initial guess')
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_InitialGuess.png'.format(numFigures, trueShiftPixels))
numFigures += 1 
plt.close() 

guessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

fillStirSpace(guessS, guessP)

recon1 = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_1.par')
poissonobj1 = recon1.get_objective_function()
poissonobj1.set_recompute_sensitivity(True)
recon1.set_up(guessS)
poissonobj1.set_recompute_sensitivity(False) 

recon2 = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_2.par')
poissonobj2 = recon2.get_objective_function()
poissonobj2.set_recompute_sensitivity(True)
recon2.set_up(guessS)
poissonobj2.set_recompute_sensitivity(False) 

quadErrorSumListList = [] 
guessPList = []
offsetFoundList = []
quadErrorSumFoundList = []

#_________________________NESTED EM LOOP_______________________________
for iIt in range(nIt):
    fillStirSpace(guessS, guessP)

    #_________________________MOTION MODEL OPTIMIZATION_______________________________
    quadErrorSumList = []

    offSets = range(trueShiftPixels/2-10,trueShiftPixels/2+11,1) # Let op: als de shift negatief is, moeten 0 en trueShiftPixels andersom staan! 

    for offset in offSets: 
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

        '''
        plt.subplot(1,3,1), plt.imshow(projectionPList[0][0,:,:]), plt.title('Guess with + offset')
        plt.subplot(1,3,2), plt.imshow(measurementListP[0][0,:,:]), plt.title('Measurement')
        plt.subplot(1,3,3), plt.imshow(abs(measurementListP[0][0,:,:]-projectionPList[0][0,:,:])), plt.title('Difference')
        plt.suptitle('Motion model optimization, offset:  {}, true shift: {}'.format(offset, trueShiftPixels))
        plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_Offset{}_Iteration{}_FirstTimeFrameProjection.png'.format(numFigures, trueShiftPixels, offset, iIt)) 
        numFigures += 1 
        plt.close() 

        plt.subplot(1,3,1), plt.imshow(projectionPList[1][0,:,:]), plt.title('Guess with - offset')
        plt.subplot(1,3,2), plt.imshow(measurementListP[1][0,:,:]), plt.title('Measurement')
        plt.subplot(1,3,3), plt.imshow(abs(measurementListP[1][0,:,:]-projectionPList[1][0,:,:])), plt.title('Difference')
        plt.suptitle('Motion model optimization, offset:  {}, true shift: {}'.format(offset, trueShiftPixels))
        plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_Offset{}_Iteration{}_SecondTimeFrameProjection.png'.format(numFigures+1, trueShiftPixels, offset, iIt))
        numFigures += 1 
        plt.close() 
        '''

    quadErrorSums = [x['quadErrorSum'] for x in quadErrorSumList]
    for i in range(len(quadErrorSumList)): 
        if(quadErrorSumList[i]['quadErrorSum'] == min(quadErrorSums)): 
            offsetFound = quadErrorSumList[i]['offset']
            offsetFoundList.append(offsetFound)
            quadErrorSumFound = quadErrorSumList[i]['quadErrorSum']
            quadErrorSumFoundList.append(quadErrorSumFound) 

    quadErrorSumListList.append(quadErrorSums)

    plt.plot(offSets, quadErrorSums, 'b-', offsetFound, quadErrorSumFound, 'ro'), plt.title('Quadratic error vs. offset')
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError_Iteration{}.png'.format(numFigures, trueShiftPixels, iIt))
    numFigures += 1 
    plt.close()

    #_________________________MOTION COMPENSATION_______________________________
    MotionModel.setOffset(offsetFound)     
    #MotionModel.setOffset(0.0) 
    recon1.reconstruct(guessS)

    MotionModel.setOffset(-offsetFound)     
    #MotionModel.setOffset(0.0) 
    recon2.reconstruct(guessS)

    reconFrame1P = stirextra.to_numpy(guessS)
    reconFrame1P[np.isnan(reconFrame1P)] = 0 # Door het verschuiven van de reconruimte schuift er een deel het beeld in waar je geen informatie over hebt, deze is leeg, om problemen te voorkomen kun je dit beter 0 maken.
    reconFrame2P = stirextra.to_numpy(guessS)
    reconFrame2P[np.isnan(reconFrame2P)] = 0

    guessP = (reconFrame2P + reconFrame1P)/2

    guessPList.append(guessP)

    plt.imshow(guessP[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Motion corrected reconstruction'), plt.axis('off')
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_OffsetFound{}_MotionCompensatedRecon_Iteration{}.png'.format(numFigures, trueShiftPixels, offsetFound, iIt))
    numFigures += 1
    plt.close()

plt.figure(figsize = (23.0, 18.0)) 
for i in range(len(guessPList)):
    plt.subplot(2, nIt/2+1, i+1), plt.title('Iteration {}'.format(i)), plt.imshow(guessPList[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.axis('off')
plt.suptitle('Motion compensated OSMAPOSL reconstruction')
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_OffsetFound{}_MotionCompensatedRecons'.format(numFigures, trueShiftPixels, offsetFound))
numFigures += 1 
plt.close()

for i in range(len(quadErrorSumListList)): 
    plt.plot(offsetFoundList, quadErrorSumFoundList, 'ro') 
    plt.plot(offSets, quadErrorSumListList[i], label = 'Iteration {}'.format(i)), plt.title('Quadratic error vs. offset')
    plt.axvline(trueShiftPixels/2, color='k', linestyle='--')
plt.legend()
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError.png'.format(numFigures, trueShiftPixels))
numFigures += 1 
plt.close()


#_________________________GUESS PERFECT SHEPP-LOGAN_______________________________
''' 
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
plt.close()

guessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
fillStirSpace(guessS, guessP)
'''



#_________________________GUESS PERFECT BLOKJE_______________________________
''' 
projection = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)

guessP = np.zeros((1, 160,160))
guessP[0, (65+trueShiftPixels/2):(95+trueShiftPixels/2), 65:95] = 1 
plt.imshow(guessP[0,:,:]), plt.title('Initial guess')
plt.savefig('./Plaatjes/Testen_shift_vinden/8_iteraties_{}_true_shift/blokjeInitialGuess.png'.format(trueShiftPixels))
plt.close()

guessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
fillStirSpace(guessS, guessP)
'''



#_________________________TEST OMSAPOSL AFTER DIFFERENT RECONS_______________________________ 
''' 
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
plt.plot(axisX, sumList, axisX, [np.sum(image)]*len(axisX)), plt.title('Sum of OSMAPOSL recon (blue), sum of original image (green)'), plt.xlabel('Iteration number')
plt.savefig('./Plaatjes/OSMAPOSLSumAfterIterations.png')
plt.close()

for i in range(8): 
    plt.subplot(2,4,i+1), plt.imshow(testList[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Iteration {0}'.format(i))
plt.savefig('./Plaatjes/OSMAPOSLReconAfterIterations.png'.format(trueShiftPixels))
plt.close() 
'''