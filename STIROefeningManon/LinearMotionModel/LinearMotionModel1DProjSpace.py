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
trueShiftAmplitude = 30 # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd) + LET OP: kan niet groter zijn dan de lengte van het plaatje (kan de code niet aan) 
numFigures = 18 
nIt = 3 # number of nested EM iterations
nFrames = 2
trueOffset = 5

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

    tmpY = np.zeros((50, np.shape(imageSmall)[1])) 
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
        shift = iFrame*trueShiftAmplitude
        tmp = np.zeros((1, Ny, Nx))
        tmp[0] = image  

        if shift > 0: 
            tmp[0, shift:Ny, :] = tmp[0, 0:(Ny-shift), :]
            tmp[0, 0:shift, :] = 0
       
        if shift < 0: 
            tmp[0, 0:(Ny+shift), :] = tmp[0, (-shift):Ny, :] 
            tmp[0, (Ny+shift):Ny, :] = 0

        phantomP.append(tmp) 
    originalImageP = phantomP[0]

    for i in range(nFrames):    
        plt.subplot(1,2,i+1), plt.title('Time frame {0}'.format(i)), plt.imshow(phantomP[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0) 
    plt.suptitle('Phantom')
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude))
    numFigures += 1 
    plt.close() 

# Sinusoidal motion
if (motion == 'Sine'):
    shiftList = [] 
    for iFrame in range(nFrames): 
        shift = int(trueShiftAmplitude * math.sin(2*math.pi*iFrame/9)) 
        shiftList.append(shift) 
        tmp = np.zeros((1, Ny, Nx))
        tmp[0] = image  
    
        if shift > 0: 
            tmp[0, shift:Ny, :] = tmp[0, 0:(Ny-shift), :]
            tmp[0, 0:shift, :] = 0
       
        if shift < 0: 
            tmp[0, 0:(Ny+shift), :] = tmp[0, (-shift):Ny, :]
            tmp[0, (Ny+shift):Ny, :] = 0

        phantomP.append(tmp) 
    originalImageP = phantomP[0]

    surSignal = [trueOffset + shiftList[i] for i in range(len(shiftList))] 

    plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Sinusoidal phantom shifts'), plt.xlabel('Time frame'), plt.ylabel('Shift')
    plt.plot(range(nFrames), shiftList, label = 'True motion')
    plt.legend(loc = 4)
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_shiftList.png'.format(numFigures, trueShiftAmplitude))
    numFigures += 1 
    plt.close()

    for i in range(nFrames):    
        plt.figure(figsize=(5.0, 5.0))
        plt.title('{0}'.format(i)), plt.imshow(phantomP[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0) 
        plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantomFrame{}.png'.format(numFigures, trueShiftAmplitude, i))
        numFigures += 1 
        plt.close() 
    
    plt.figure(figsize=(23.0, 21.0))
    for i in range(nFrames):    
        plt.subplot(2,nFrames/2+1,i+1), plt.title('{0}'.format(i)), plt.imshow(phantomP[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0) 
    plt.suptitle('Phantom')
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude))
    numFigures += 1 
    plt.close() 


# STIR 
originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(originalImageS, originalImageP)

phantomS = []
for i in range(nFrames): 
    imageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
    fillStirSpace(imageS, phantomP[i])
    phantomS.append(imageS)
    del(imageS) 


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
MotionModel.setOffset(0.0) 

for i in range(nFrames):     
    forwardprojector.forward_project(measurement, phantomS[i])
    measurement.write_to_file('sinoMeas_{}.hs'.format(i+1))
    measurementS = measurement.get_segment_by_sinogram(0)
    measurementP = stirextra.to_numpy(measurementS)
    if (noise): 
        measurementP = sp.random.poisson(measurementP)
    measurementListP.append(measurementP) 
    plt.subplot(2,nFrames/2+1,i+1), plt.imshow(measurementP[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Meas. TF0')

plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_Measurements.png'.format(numFigures, trueShiftAmplitude)) 
numFigures += 1 
plt.close()


#_________________________CONFIG FILES_______________________________ 
for i in range(nFrames): 
    file = open('config_Proj_{}.par'.format(i+1), 'w')
    file.write('OSMAPOSLParameters := \n') 
    file.write('objective function type:= PoissonLogLikelihoodWithLinearModelForMeanAndProjData \n') 
    file.write('PoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters:= \n')
    file.write('input file := sinoMeas_{}.hs \n'.format(i+1))
    file.write('end PoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters:= \n')
    file.write('initial estimate:= some_image \n')
    file.write('output filename prefix := output_config_Proj_{} \n'.format(i+1))
    file.write('number of subsets:= 1 \n')
    file.write('number of subiterations:= 1 \n')
    file.write('Save estimates at subiteration intervals:= 1 \n')
    file.write('END := \n')
    file.close() 


#_________________________GUESS_______________________________
#Negeer voor nu het initial estimate
projection = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
MotionModel.setOffset(0.0)
initialGuessPList = []
for i in range(nFrames): 
    reconGuessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                        stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                        stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
    reconGuessS.fill(1) # moet er staan

    recon = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_{}.par'.format(i+1)) 
    recon.set_up(reconGuessS)
    recon.reconstruct(reconGuessS)
    initialGuessPtmp = stirextra.to_numpy(reconGuessS)
    initialGuessPList.append(initialGuessPtmp) 

guessP = np.mean(initialGuessPList, axis = 0)
plt.imshow(guessP[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Initial guess')
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close() 

guessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
fillStirSpace(guessS, guessP)

reconList = []
for i in range(nFrames): 
    recon = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_{}.par'.format(i+1))
    poissonobj = recon.get_objective_function()
    poissonobj.set_recompute_sensitivity(True)
    recon.set_up(guessS)
    poissonobj.set_recompute_sensitivity(False) 
    reconList.append(recon) 

quadErrorSumListList = [] 
guessPList = []
offsetFoundList = []
quadErrorSumFoundList = []

guessPList.append(guessP)


#_________________________NESTED EM LOOP_______________________________
for iIt in range(nIt):
    fillStirSpace(guessS, guessP)

    #_________________________MOTION MODEL OPTIMIZATION_______________________________
    quadErrorSumList = []

    offSets = range(trueShiftAmplitude/2-10,trueShiftAmplitude/2+11,1) 

    for offset in offSets: 
        projectionPList = []

        surTMP = [1, -1] 
        for iFrame in range(nFrames): 
            MotionModel.setOffset(surTMP[iFrame]*offset) # Is this also the right sign if the real shift is negative? 
            forwardprojector.forward_project(projection, guessS)
            projection.write_to_file('sino_{}.hs'.format(iFrame+1))
            projectionS = projection.get_segment_by_sinogram(0)
            projectionP = stirextra.to_numpy(projectionS)
            projectionPList.append(projectionP)

        quadErrorSum = 0 
        for iFrame in range(nFrames): 
            quadErrorSum += np.sum((projectionPList[iFrame][0,:,:] - measurementListP[iFrame][0,:,:])**2)
    
        quadErrorSumList.append({'offset' : offset, 'quadErrorSum' : quadErrorSum})

    quadErrorSums = [x['quadErrorSum'] for x in quadErrorSumList]
    for i in range(len(quadErrorSumList)): 
        if(quadErrorSumList[i]['quadErrorSum'] == min(quadErrorSums)): 
            offsetFound = quadErrorSumList[i]['offset']
            offsetFoundList.append(offsetFound)
            quadErrorSumFound = quadErrorSumList[i]['quadErrorSum']
            quadErrorSumFoundList.append(quadErrorSumFound) 

    quadErrorSumListList.append(quadErrorSums)

    plt.plot(offSets, quadErrorSums, 'b-', offsetFound, quadErrorSumFound, 'ro'), plt.title('Quadratic error vs. offset')
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError_Iteration{}.png'.format(numFigures, trueShiftAmplitude, iIt))
    numFigures += 1 
    plt.close()

    #_________________________MOTION COMPENSATION_______________________________
    # surSignal = [0, 1] 
    # offsetFound = 24 # offsetFound na de eerste iteratie was 12, dus dit werkt blijkbaar even goed! 
    for iFrame in range(nFrames): 
        MotionModel.setOffset(surTMP[iFrame]*offsetFound) 
        reconList[iFrame].reconstruct(guessS)
    
    reconFramePList = []
    for iFrame in range(nFrames): 
        reconFrameP = stirextra.to_numpy(guessS)
        reconFrameP[np.isnan(reconFrameP)] = 0
        reconFramePList.append(reconFrameP) 

    guessP = np.zeros(np.shape(originalImageP))
    for iFrame in range(nFrames): 
        guessP += reconFramePList[iFrame]
    guessP /= len(reconFramePList)

    #guessP = (reconFramePList[0] + reconFramePList[1])/2

    guessPList.append(guessP)

    plt.imshow(guessP[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Motion corrected reconstruction, offset: {}'.format(offsetFound)), plt.axis('off')
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_MotionCompensatedRecon_Iteration{}.png'.format(numFigures, trueShiftAmplitude, iIt))
    numFigures += 1
    plt.close()

plt.figure(figsize = (23.0, 18.0)) 
for i in range(len(guessPList)):
    plt.subplot(2, nIt/2+1, i+1), plt.title('Iteration {}'.format(i)), plt.imshow(guessPList[i][0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.axis('off')
plt.suptitle('Motion compensated OSMAPOSL reconstruction, offset: {}'.format(offsetFound))
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_MotionCompensatedRecons'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close()

for i in range(len(quadErrorSumListList)): 
    plt.plot(offsetFoundList, quadErrorSumFoundList, 'ro') 
    plt.plot(offSets, quadErrorSumListList[i], label = 'Iteration {}'.format(i)), plt.title('Quadratic error vs. offset')
    plt.axvline(0.5*shiftList[1], color='k', linestyle='--')
plt.legend()
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close()


#_________________________GUESS PERFECT SHEPP-LOGAN_______________________________
''' 
## 
projection = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)

shift = int(0.5*iFrame*trueShiftAmplitude)
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
plt.savefig('./Plaatjes/Testen_shift_vinden/8_iteraties_{}_true_shift/perfectInitialGuess.png'.format(trueShiftAmplitude))
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
guessP[0, (65+trueShiftAmplitude/2):(95+trueShiftAmplitude/2), 65:95] = 1 
plt.imshow(guessP[0,:,:]), plt.title('Initial guess')
plt.savefig('./Plaatjes/Testen_shift_vinden/8_iteraties_{}_true_shift/blokjeInitialGuess.png'.format(trueShiftAmplitude))
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
plt.savefig('./Plaatjes/OSMAPOSLReconAfterIterations.png'.format(trueShiftAmplitude))
plt.close() 
'''