import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon, rescale
import ManonsFunctions as mf 
import scipy as sp
import pyvpx
import copy
from scipy.optimize import curve_fit, minimize, brute
from scipy.signal import argrelextrema
from heapq import merge
from matplotlib.ticker import MaxNLocator
from skimage.measure import find_contours, points_in_poly

#_________________________PARAMETER SETTINGS_______________________________

stationary = False 
#stationary = False # False is only possible for sinusoidal motion! 
modelBroken = True  
#modelBroken = False  

# Parameters that influence the figure saving directory 
phantom = 'Liver'
#phantom = 'Block'
#noise = False
noise = True
#motion = 'Step' 
motion = 'Sine'

# Create a direcotory for figure storage (just the string, make sure  the folder already exists!) 
dir = './Figures/'
figSaveDir = mf.make_figSaveDir(dir, motion, phantom, noise, stationary, modelBroken)

# Parameters that do not influence the saving directory 
nIt = 10
nModelSkip = 3
trueShiftAmplitude = 5 # Make sure this is not too large, activity moving out of the FOV will cause problems 
trueSlope = 1.4 # y-axis 
trueSlopeX = 0.2 # x-axis 
numFigures = 0 
if (motion == 'Step'): nFrames = 2 
else: nFrames = 18
noiseLevel = 200
x0 = np.array([1.0, 1.0]) # initial guess for the optimization function 
mu = 9.687E-2 # water, cm^-1
mu /= 5 # number of pixels in 1 cm if one pixel is 2 mm 

iAngles = np.linspace(0, 180, 60)[:-1]

# Store all settings in a text file 
mf.write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueSlope, nFrames, x0, modelBroken)


#_________________________MAKE PHANTOM_______________________________
# Make phantom 
image2D = mf.make_Phantom(phantom, noiseLevel)

nX = int(np.shape(image2D)[0]) # Not sure if x and y are actually x and y 
nY = int(np.shape(image2D)[1])
imageMuBinary = copy.deepcopy(image2D)
for iX in range(nX): 
    for iY in range(nY): 
        if imageMuBinary[iX, iY] == 0: continue 
        else: imageMuBinary[iX, iY] /= image2D[iX, iY] 
imageMu = imageMuBinary*mu
sinoMu = radon(copy.deepcopy(imageMu), iAngles) 
expSinoMu = np.zeros(np.shape(sinoMu))
for i in range(np.shape(sinoMu)[0]):
    for j in range(np.shape(sinoMu)[1]):
        expSinoMu[i,j] = math.exp(-sinoMu[i,j])

# Plot phantom and store as vpx image 
plt.figure(), plt.title('Original image'), plt.imshow(image2D, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_phantom.png'.format(numFigures)), plt.close()
numFigures += 1

image2DTMP = np.zeros((1,) + np.shape(image2D) )
image2DTMP[0,:,:] = image2D
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'OriginalImage.vpx') 


#_________________________ADD MOTION_______________________________ 
# Create surrogate signal and add motion to the phantom  
phantomList, surSignal, shiftList, shiftXList = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueSlope, trueSlopeX, image2D, stationary, modelBroken)
originalImage = phantomList[0]

# Plot surrogate signal and internal motion 
# y-axis
plt.figure()
plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Motion (y-axis)'), plt.xlabel('Time frame'), plt.ylabel('Shift')
plt.plot(range(nFrames), shiftList, label = 'True motion y-axis')
plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_shiftList.png'.format(numFigures)), plt.close()
numFigures += 1 


#_________________________MEASUREMENT, INITIAL GUESS, NORMALIZATION_______________________________

# Create sinograms of each frame and add Poisson noise to them 
measList = []
for iFrame in range(nFrames):
    meas = radon(copy.deepcopy(phantomList[iFrame])[0,:,:], iAngles)
    if (iFrame == 0): measNoNoise = meas
    if (noise): meas = sp.random.poisson(meas)
    if (iFrame == 0): measWithNoise = meas
    meas = meas.astype(np.float) * expSinoMu 
    if (iFrame == 0): measAC = meas 
    measList.append(meas.astype(np.float)) 

# Plot sinogram of time frame 0 with and without noise  
plt.figure() 
plt.subplot(1,3,1), plt.title('Without noise'), plt.imshow(measNoNoise, interpolation=None, vmin = 0, vmax = np.max(measWithNoise), cmap=plt.cm.Greys_r)
plt.subplot(1,3,2), plt.title('With noise'), plt.imshow(measWithNoise, interpolation=None, vmin = 0, vmax = np.max(measWithNoise), cmap=plt.cm.Greys_r)
plt.subplot(1,3,3), plt.title('With AC'), plt.imshow(measAC, interpolation=None, vmin = 0, vmax = np.max(measWithNoise), cmap=plt.cm.Greys_r)
plt.suptitle('Time Frame 1'), plt.savefig(figSaveDir + 'Fig{}_measurementsWithWithoutNoise.png'.format(numFigures)), plt.close()
numFigures += 1 

# Initial guess - image 
guess = np.ones(np.shape(image2D)) # Fills it with floats, not ints 

# Objective function for model optimization
def computeQuadError(x, nFrames, guess, surSignal, iAngles, returnTuple, negNumFile):    
    quadErrorSum = 0.0
    quadErrorSumList = []
    
    for iFrame in range(nFrames): 
        guessMoved = np.zeros(np.shape(guess))
        guessMoved = sp.ndimage.shift(copy.deepcopy(guess), (surSignal[iFrame] * x[0], surSignal[iFrame] * x[1])) 
        if (np.min(guessMoved) < 0): negNumFile.write('computeQuadError, guessMoved {}\n'.format(np.min(guessMoved)))
        guessMovedProj = radon(copy.deepcopy(guessMoved), iAngles)
        diff = guessMovedProj/np.sum(guessMovedProj) - measList[iFrame]/np.sum(measList[iFrame])
        quadError = np.sum(np.abs(diff)) 
        quadError /= 2 # Now 0 <= quadError <= 1 
        quadErrorSum += quadError
        quadErrorSumList.append(quadError)

    quadErrorSum /= nFrames # Now 0 <= quadErrorSum <= 1 

    if (returnTuple): return quadErrorSumList 
    else: return quadErrorSum


#_________________________NESTED EM LOOP_______________________________
slopeFound = 0.0 # the first MLEM iterations are regular (image is not shifted/corrected) 
slopeXFound = 0.0 

slopeList = np.linspace(1, 2, 9)

parFile = open(figSaveDir + "Parameters.txt", "w")
negNumFile = open(figSaveDir + "NegativeNumbers.txt", "w")

# Lists for storage 
quadErrorsList = [] ### 
slopeFoundList = []
slopeXFoundList = []
quadErrorFoundList = []
guessSum = []
guessSum.append(np.sum(guess))
for iIt in range(nIt): 
    # Motion model optimization
    if (iIt >= nModelSkip):
        # Rough estimate of the variation in quadratic error as a function of the slope (of the y-axis) ### 
        quadErrors = [computeQuadError(np.array([i, slopeXFound]), nFrames, guess, surSignal, iAngles, False, negNumFile) for i in slopeList] ### 
        quadErrorsList.append(quadErrors) ### 

        plt.figure()         
        plt.plot(slopeList, quadErrors), plt.title('Parameter space iteration {}'.format(iIt+1))
        plt.savefig(figSaveDir + 'Fig{}_ParSpaceSampling.png'.format(numFigures))
        numFigures += 1 
        plt.close() 

        args = (nFrames, guess, surSignal, iAngles, False, negNumFile)
        myOptions = {'disp': True, 'maxiter' : 10}
        res = minimize(computeQuadError, x0, args, method = 'Powell', options = myOptions)
        slopeFound = res.x[0]        
        slopeFoundList.append(slopeFound)
        slopeXFound = res.x[1]  
        slopeXFoundList.append(slopeXFound) 
        quadErrorFound = res.fun
        quadErrorFoundList.append(quadErrorFound)  

        # Time-resolved quadratic error 
        quadErrorSumList = computeQuadError((slopeFound, slopeXFound), nFrames, guess, surSignal, iAngles, True, negNumFile)     

        plt.figure() 
        plt.plot(quadErrorSumList), plt.title('Normalized error vs. time, iteration {}'.format(iIt+1))
        plt.axis([0.0, nFrames, 0.0, 1.0])
        plt.savefig(figSaveDir + 'Fig{}_QuadraticError_Time.png'.format(numFigures))
        numFigures += 1 
        plt.close() 
        
        print res.message 
        print 'Slope found: {}'.format(slopeFound)
        print 'SlopeX found: {}'.format(slopeXFound)
        parFile.write('Iteration {}\n'.format(iIt+1))
        parFile.write('objective function: {}\n'.format(res.fun))
        parFile.write('slope: {}\n'.format(slopeFound)) 
        parFile.write('slopeX: {}\n\n'.format(slopeXFound)) 

    totalError = 0 
    # MLEM with motion compensation 
    for iFrame in range(nFrames): 
        # Shift guess for the current model, in time frame iFrame, and forward project it 
        shiftedGuess = np.zeros(np.shape(guess)) 
        shiftedGuess = sp.ndimage.shift(copy.deepcopy(guess), (surSignal[iFrame] * slopeFound, surSignal[iFrame] * slopeXFound)) 
        if (np.min(shiftedGuess) < 0): negNumFile.write('MLEM, shiftedGuess: {}\n'.format(np.min(shiftedGuess)))
        shiftedGuessSinogram = radon(shiftedGuess, iAngles) 

        # Compute error between measured sinogram and guess
        error = measList[iFrame]/shiftedGuessSinogram 
        error[np.isnan(error)] = 0
        error[np.isinf(error)] = 0
        error[error > 1E10] = 0
        error[error < 1E-10] = 0

        # Backproject error and shift back 
        errorBck = iradon(error, iAngles, filter = None) 
        errorBckShifted = np.zeros(np.shape(errorBck)) 
        errorBckShifted = sp.ndimage.shift(errorBck, (-surSignal[iFrame] * slopeFound, -surSignal[iFrame] * slopeXFound))
        if (np.min(errorBckShifted) < 0): negNumFile.write('MLEM, errorBckShifted: {}\n'.format(np.min(errorBckShifted)))

        # Update total error 
        totalError += errorBckShifted   
    
    # Update guess with the error from all time frames
    guess *= totalError/nFrames

    # Normalization 
    guess /= np.sum(guess) 
    guess *= np.sum(measList[-1])/np.shape(measList[-1])[1] 

    guessSum.append(np.sum(guess))
    countIt = iIt+1 

    # Plot current guess 
    plt.figure(), plt.title('Guess after {} iteration(s)'.format(iIt+1)), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_finalImage.png'.format(numFigures)), plt.close()
    numFigures += 1  
    guessTMP = np.zeros((1,) + np.shape(guess))
    guessTMP[0,:,:] = guess
    pyvpx.numpy2vpx(guessTMP, figSaveDir + 'guess_Iteration{}.vpx'.format(iIt)) 

parFile.close() 
negNumFile.close() 

# Plot and save original image and reconstructed image 
plt.figure(), plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(originalImage[0,:,:], interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('Reconstructed Image'), plt.imshow(guess, interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_originalAndRecon.png'.format(numFigures)), plt.close() 
numFigures += 1 

# Plot quadratic errors of all iteqrations
# y-axis
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # to get integer values on the x-axis
if (not modelBroken): 
    plt.axhline(trueSlope, color = 'k', linestyle = '--', label = 'Correct value')
else: 
    plt.axhline(trueSlope, color = 'k', linestyle = '--', label = 'Correct value 1st half')
    plt.axhline(0.8, color = 'k', linestyle = '-.', label = 'Correct value 2nd half')
plt.plot(range(nModelSkip+1, nIt+1), slopeFoundList, 'ro', label = 'Estimated value') 
plt.title('Parameter optimization (y-axis)'), plt.xlabel('Iteration number'), plt.ylabel('Slope')
plt.legend(), plt.savefig(figSaveDir + 'Fig{}_SlopesFoundY.png'.format(numFigures))
numFigures += 1 
plt.close()

# x-axis
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # to get integer values on the x-axis
plt.axhline(trueSlopeX, color = 'k', linestyle = '--', label = 'Correct value')
plt.plot(range(nModelSkip+1, nIt+1), slopeXFoundList, 'ro', label = 'Estimated value') 
plt.title('Parameter optimization (x-axis)'), plt.xlabel('Iteration number'), plt.ylabel('Slope')
plt.legend(), plt.savefig(figSaveDir + 'Fig{}_SlopesFoundX.png'.format(numFigures))
numFigures += 1 
plt.close()




'''
smallVolume[27:30,34:37] = 0 
smallContourReversed = find_contours(smallVolume, 0) 
smallContour = [(i[1], i[0]) for i in smallContourReversed[0]]
smallContourX, smallContourY = np.array(smallContour).T
plt.imshow(smallVolume), plt.scatter(smallContourX, smallContourY), plt.show() 

smallNx = np.shape(smallVolume)[1]
smallNy = np.shape(smallVolume)[0]
smallTargetCoordsMatrix = [[(i,j) for i in range(smallNx)] for j in range(smallNy)] 
smallTargetCoordsVec = []
for i in range(len(smallTargetCoordsMatrix)): smallTargetCoordsVec += smallTargetCoordsMatrix[i]

smallBinMaskVec = points_in_poly(smallTargetCoordsVec, smallContour)

smallBinMaskMatrix = np.zeros(np.shape(smallVolume))
for i in range(smallNx): 
    for j in range(smallNy): 
        if(smallBinMaskVec[i*(smallNx-1) + i + 1]): smallBinMaskMatrix[i,j] = 1 

nX = 5 
nY = 3 
testMatrix = np.zeros((nY,nX))
testCoordsMatrix = [[(i,j) for i in range(nX)] for j in range(nY)] 
testCoordsVec = [] 
for i in range(len(testCoordsMatrix)): testCoordsVec += testCoordsMatrix[i]

smallTargetCoordsVec = []
for i in range(len(smallTargetCoordsMatrix)): smallTargetCoordsVec += smallTargetCoordsMatrix[i]
binMaskVec = [False, False, True, False, False, True, False, False, True, False, False, True, False, False, True]   
binMaskMatrix = np.zeros(np.shape(smallTargetVolume))
for i in range(nX): 
    for j in range(nY): 
        if(binMaskVec[j*(nX-1)+i+1]): binMaskMatrix[i,j] = 1 
'''