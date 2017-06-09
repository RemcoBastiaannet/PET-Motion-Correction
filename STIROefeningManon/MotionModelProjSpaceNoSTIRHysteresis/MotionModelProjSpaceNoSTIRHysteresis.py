import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon
import ManonsFunctionsHysteresis as mf 
import scipy as sp
import pyvpx
import copy
from scipy.optimize import curve_fit, minimize, brute
from scipy.signal import argrelextrema
from heapq import merge


#_________________________PARAMETER SETTINGS_______________________________
# Parameters that influence the figure saving directory 
phantom = 'Shepp-Logan'
#noise = False
noise = True
#motion = 'Step' 
motion = 'Sine'
#stationary = True 
stationary = False # False is only possible for sinusoidal motion! 
hysteresis = False 
#hysteresis = True
drift = True
#drift = False

# Create a direcotory for figure storage (just the string, make sure  the folder already exists!) 
dir = './Figures/'
figSaveDir = mf.make_figSaveDir(dir, motion, phantom, noise, stationary)

# Parameters that do not influence the saving directory 
nIt = 13
trueShiftAmplitude = 10 # Make sure this is not too large, activity moving out of the FOV will cause problems 
trueSlope = 0.5 # y-axis 
trueSlopeX = 0.2 # x-axis 
trueSlopeInhale = 1.0 # hysteresis, x-axis
trueSlopeExhale = trueSlopeInhale # hysteresis, x-axis, must be the same as trueSlopeInhale, otherwise the two functions do are not equal at the endpoints
trueSquareSlopeInhale = +0.1 # hysteresis, x-axis
trueSquareSlopeExhale = -0.06 # hysteresis, x-axis
numFigures = 0 
if (motion == 'Step'): nFrames = 2 
else: nFrames = 36
noiseLevel = 600
x0 = np.array([1.0, 1.0]) # initial guess for the optimization function 

# Store all settings in a text file 
mf.write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueSlope, trueSlopeInhale, trueSlopeExhale, trueSquareSlopeInhale, trueSquareSlopeExhale, nFrames, hysteresis, x0)


#_________________________MAKE PHANTOM_______________________________
# Make phantom 
image2D = mf.make_Phantom(phantom, noiseLevel)

# Plot phantom and store as vpx image 
plt.figure(), plt.title('Original image'), plt.imshow(image2D, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1

image2DTMP = np.zeros((1,) + np.shape(image2D) )
image2DTMP[0,:,:] = image2D
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'OriginalImage.vpx') 
 

#_________________________ADD MOTION_______________________________ 
# Create surrogate signal and add motion to the phantom  
phantomList, surSignal, shiftList, shiftXList = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueSlope, trueSlopeX, trueSlopeInhale, trueSlopeExhale, trueSquareSlopeInhale, trueSquareSlopeExhale, image2D, stationary, hysteresis)
originalImage = phantomList[0]

# Plot hysteresis on x-axis
plt.figure() 
plt.plot(surSignal, shiftXList, 'bo', markersize = 4.0), plt.title('Hysteresis (x-axis)'), plt.xlabel('Surrogate signal (external motion)'), plt.ylabel('Internal motion x-axis')
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_Hysteresis.png'.format(numFigures, trueShiftAmplitude)), plt.close()
plt.show()
numFigures += 1 

# Plot hysteresis on y-axis
plt.figure() 
plt.plot(surSignal, shiftList, 'bo', markersize = 4.0), plt.title('Hysteresis (y-axis)'), plt.xlabel('Surrogate signal (external motion)'), plt.ylabel('Internal motion y-axis')
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_Hysteresis.png'.format(numFigures, trueShiftAmplitude)), plt.close()
plt.show()
numFigures += 1 

'''
# Plot a time series of the phantom 
for iFrame in range(nFrames):    
    plt.title('Time frame {0}'.format(iFrame)), plt.imshow(phantomList[iFrame][0,:,:], interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r)
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom_TF{}.png'.format(numFigures, trueShiftAmplitude, iFrame)), plt.close()
    numFigures += 1 
''' 

#_________________________DISTINGUISH INHALE AND EXHALE PHASES_______________________________ 
# Derivatives, the sign of which distinguishes between inhale and exhale 
if (hysteresis): 
    surSignalDiff = np.diff(np.array(surSignal))
    shiftXListDiff = np.diff(np.array(shiftXList))

    # Lists for storage 
    inhaleSurSignal = [] 
    inhaleShiftXList = []
    inhaleSurAxis = []
    inhaleShiftXAxis = [] 
    exhaleSurSignal = [] 
    exhaleShiftXList = [] 
    exhaleSurAxis = [] 
    exhaleShiftXAxis = [] 

    # Distinguish inhale and exhale phases 
    for i in range(len(surSignalDiff)): 
        if (surSignalDiff[i] > 0): 
            inhaleSurSignal.append(surSignal[i])
            inhaleSurAxis.append(i)
        else: 
            exhaleSurSignal.append(surSignal[i]) 
            exhaleSurAxis.append(i)
        if (shiftXListDiff[i] > 0): 
            inhaleShiftXList.append(shiftXList[i])          
            inhaleShiftXAxis.append(i)
        else: 
            exhaleShiftXList.append(shiftXList[i]) 
            exhaleShiftXAxis.append(i)

# Plot surrogate signal and internal motion 
# y-axis
plt.figure()
plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Motion (y-axis)'), plt.xlabel('Time frame'), plt.ylabel('Shift')
plt.plot(range(nFrames), shiftList, label = 'True motion y-axis'), plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_shiftList.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

if (hysteresis): 
    # x-axis, inhale
    plt.figure() 
    plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Motion (y-axis, inhale)'), plt.xlabel('Time frame'), plt.ylabel('Shift')
    plt.plot(inhaleSurAxis, inhaleSurSignal, 'ro', label = 'Inhale surrogate') 
    plt.plot(range(nFrames), shiftXList, label = 'Internal motion')
    plt.plot(inhaleShiftXAxis, inhaleShiftXList, 'ro') 
    plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_Inhale.png'.format(numFigures, trueShiftAmplitude)), plt.close()
    numFigures += 1 
    # x-axis, exhale
    plt.figure() 
    plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Motion (y-axis, exhale)'), plt.xlabel('Time frame'), plt.ylabel('Shift')
    plt.plot(exhaleSurAxis, exhaleSurSignal, 'go', label = 'Exhale') 
    plt.plot(range(nFrames), shiftXList, label = 'Internal motion')
    plt.plot(exhaleShiftXAxis, exhaleShiftXList, 'go') 
    plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_Exhale.png'.format(numFigures, trueShiftAmplitude)), plt.close()
    numFigures += 1 


#_________________________MEASUREMENT, INITIAL GUESS, NORMALIZATION_______________________________
# Angles for randon
iAngles = np.linspace(0, 360, 120)[:-1]

# Create sinograms of each frame and add Poisson noise to them 
measList = []
for iFrame in range(nFrames):
    meas = radon(copy.deepcopy(phantomList[iFrame])[0,:,:], iAngles) 
    if (iFrame == 0): measNoNoise = meas
    if (noise): meas = sp.random.poisson(meas)
    if (iFrame == 0): measWithNoise = meas
    measList.append(meas) 

# Plot sinogram of time frame 0 with and without noise  
plt.figure() 
plt.subplot(1,2,1), plt.title('Without noise'), plt.imshow(measNoNoise, interpolation=None, vmin = 0, vmax = np.max(measWithNoise), cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('With noise'), plt.imshow(measWithNoise, interpolation=None, vmin = 0, vmax = np.max(measWithNoise), cmap=plt.cm.Greys_r)
plt.suptitle('Time Frame 1'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_measurementsWithWithoutNoise.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

# Initial guess - image 
guess = np.ones(np.shape(image2D))

# Plot and save initial guess 
plt.figure(), plt.title('Initial guess'), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(guess), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 
guessTMP = np.zeros((1,) + np.shape(image2D))
guessTMP[0,:,:] = guess
pyvpx.numpy2vpx(guessTMP, figSaveDir + 'guess.vpx') 
    

#_________________________NESTED EM LOOP_______________________________
# Objective function for model optimization
def computeQuadError(x, nFrames, guess, surSignal, iAngles, returnTuple):    
    quadErrorSum = 0.0
    quadErrorSumList = []
    
    for iFrame in range(nFrames): 
        guessMoved = np.zeros(np.shape(guess))
        guessMoved = sp.ndimage.shift(copy.deepcopy(guess), (surSignal[iFrame] * x[0], 0)) 
        guessMoved = sp.ndimage.shift(copy.deepcopy(guessMoved), (0, surSignal[iFrame] * x[1])) 
        guessMovedProj = radon(copy.deepcopy(guessMoved), iAngles)
        quadError = np.sum((guessMovedProj/np.sum(guessMovedProj) - measList[iFrame]/np.sum(measList[iFrame]))**2)
        quadErrorSum += quadError
        quadErrorSumList.append(quadError)

    if (returnTuple): return quadErrorSumList 
    else: return quadErrorSum

slopeFound = 0.0 # the first MLEM iterations are regular (image is not shifted/corrected) 
slopeXFound = 0.0 

slopeList = np.linspace(-1, 2, 9)

parFile = open(figSaveDir + "Parameters.txt", "w")

# Lists for storage 
#quadErrorsList = []
slopeFoundList = []
slopeXFoundList = []
quadErrorFoundList = []
guessSum = []
guessSum.append(np.sum(guess))
for iIt in range(nIt): 
    # Motion model optimization
    if (iIt >= 3): 
        #quadErrors = [computeQuadError(np.array([i, trueSlopeX]), nFrames, guess, surSignal, iAngles) for i in slopeList]
        #quadErrorsList.append(quadErrors)

        '''
        args = (nFrames, guess, surSignal, iAngles, False)
        res = minimize(computeQuadError, x0, args, method = 'BFGS', options = {'disp': True, 'maxiter' : 10})
        slopeFound = res.x[0]        
        slopeFoundList.append(slopeFound)
        slopeXFound = res.x[1]  
        slopeXFoundList.append(slopeXFound) 
        quadErrorFound = res.fun
        quadErrorFoundList.append(quadErrorFound)
        '''    

        quadErrorSumList = computeQuadError((0, 0), nFrames, guess, surSignal, iAngles, True)   
        # Moving average window 
        windowLength = 10 
        window = np.ones(windowLength,'d')
        quadErrorSumListAVG = np.convolve(window/window.sum(), quadErrorSumList, mode='same')

        plt.figure() 
        plt.plot(quadErrorSumList), plt.title('Quadratic error vs. time, iteration {}'.format(iIt+1))
        plt.plot(quadErrorSumListAVG, label = 'Moving average'), 
        diffTMP = np.max(quadErrorSumList) - np.min(quadErrorSumList)
        plt.axis([0, nFrames, np.min(quadErrorSumList) - 0.1*diffTMP, np.max(quadErrorSumList) + 0.1*diffTMP])
        plt.legend() 
        plt.savefig(figSaveDir + 'Fig{}_QuadraticError_Time.png'.format(numFigures))
        numFigures += 1 
        plt.close() 

        print 'Slope found: {}'.format(slopeFound)
        print 'SlopeX found: {}'.format(slopeXFound)
        parFile.write('Iteration {}\n'.format(iIt+1))
        parFile.write('objective function: {}\n'.format(res.fun))
        parFile.write('slope: {}\n'.format(slopeFound)) 
        parFile.write('slopeX: {}\n\n'.format(slopeXFound)) 

        #plt.plot(slopeList, quadErrors, 'b-', label = ''), plt.title('Quadratic error vs. slope, iteration {}'.format(iIt+1))
        plt.plot(slopeFound, quadErrorFound, 'ro', label = 'Estimated value')
        plt.axvline(trueSlope, color='k', linestyle='--', label = 'Correct  value')
        plt.legend()
        plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError_Iteration{}.png'.format(numFigures, trueShiftAmplitude, iIt))
        numFigures += 1 
        plt.close()

    totalError = 0 
    # MLEM with motion compensation 
    for iFrame in range(nFrames): 
        # Shift guess for the current model, in time frame iFrame, and forward project it 
        shiftedGuess = np.zeros(np.shape(guess)) 
        shiftedGuess = sp.ndimage.shift(copy.deepcopy(guess), (surSignal[iFrame] * slopeFound, 0)) 
        shiftedGuess = sp.ndimage.shift(copy.deepcopy(shiftedGuess), (0, surSignal[iFrame] * slopeXFound)) 
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
        errorBckShifted = sp.ndimage.shift(errorBck, (-surSignal[iFrame] * slopeFound, 0))
        errorBckShifted = sp.ndimage.shift(errorBckShifted, (0, -surSignal[iFrame] * slopeXFound))

        # Update total error 
        totalError += errorBckShifted   
    
    # Update guess with the error from all time frames
    guess *= totalError/nFrames

    # Normalization 
    guess /= np.sum(guess) 
    guess *= np.sum(measList[-1])/np.shape(measList[-1])[1] 

    guessSum.append(np.sum(guess))
    countIt = iIt+1 

    # Save and plot current guess 
    guessTMP = np.zeros((1,) + np.shape(image2D))
    guessTMP[0,:,:] = guess
    pyvpx.numpy2vpx(guessTMP, figSaveDir + 'guess_{}.vpx'.format(iIt)) 
    plt.figure(), plt.title('Guess after {} iteration(s)'.format(iIt+1)), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_finalImage.png'.format(numFigures, trueShiftAmplitude)), plt.close()
    numFigures += 1  

parFile.close() 

# Plot original image and reconstructed image 
plt.figure(), plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(originalImage[0,:,:], interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('Reconstructed Image'), plt.imshow(guess, interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_originalAndRecon.png'.format(numFigures, trueShiftAmplitude)), plt.close() 
numFigures += 1 

# Plot some of guess as a function of iteration number 
plt.figure() 
plt.plot(guessSum), plt.title('Sum of guess'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_guessSum.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close() 

# Plot quadratic errors of all iterations
for i in range(len(slopeFoundList)): 
    if (i == 0): plt.plot(slopeFoundList, quadErrorFoundList, 'ro', label = 'Estimated value') 
    else: plt.plot(slopeFoundList, quadErrorFoundList, 'ro') 
    if (i == 0): plt.axvline(trueSlope, color='k', linestyle='--', label = 'Correct value')
    else: plt.axvline(trueSlope, color='k', linestyle='--')
    plt.title('Quadratic error vs. slope (y-axis)'), plt.xlabel('Quadratic error'), plt.ylabel('Slope')
   # plt.plot(slopeList, quadErrorsList[i], label = 'Iteration {}'.format(i+1)), plt.title('Quadratic error vs. slope')
plt.legend()
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticErrorY.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close()

for i in range(len(slopeXFoundList)): 
    if (i == 0): plt.plot(slopeXFoundList, quadErrorFoundList, 'ro', label = 'Estimated value') 
    else: plt.plot(slopeXFoundList, quadErrorFoundList, 'ro') 
    if (i == 0): plt.axvline(trueSlopeX, color='k', linestyle='--', label = 'Correct value')
    else: plt.axvline(trueSlopeX, color='k', linestyle='--')
    plt.title('Quadratic error vs. slope (x-axis)'), plt.xlabel('Quadratic error'), plt.ylabel('Slope')
   # plt.plot(slopeXList, quadErrorsList[i], label = 'Iteration {}'.format(i+1)), plt.title('Quadratic error vs. slope')
plt.legend()
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticErrorX.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close()