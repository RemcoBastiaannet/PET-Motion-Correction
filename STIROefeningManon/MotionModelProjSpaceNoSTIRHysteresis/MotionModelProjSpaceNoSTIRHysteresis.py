import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon
import ManonsFunctionsHysteresis as mf 
import scipy as sp
import pyvpx
import copy
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from heapq import merge


#_________________________PARAMETER SETTINGS_______________________________
# Parameters that influence the figure saving directory 
phantom = 'Shepp-Logan'
#noise = False
noise = True
#motion = 'Step' 
motion = 'Sine'
stationary = True 
#stationary = False # False is only possible for sinusoidal motion! 

# Create a direcotory for figure storage (just the string, make sure  the folder already exists!) 
dir = './Figures/'
figSaveDir = mf.make_figSaveDir(dir, motion, phantom, noise, stationary)

# Parameters that do not influence the saving directory 
nIt = 10 
trueShiftAmplitude = 10 # Make sure this is not too large, activity moving out of the FOV will cause problems 
trueSlope = 0.5 # y-axis 
trueSlopeInhale = 1.0 # x-axis
trueSlopeExhale = trueSlopeInhale # x-axis, must be the same as trueSlopeInhale, otherwise the two functions do are not equal at the endpoints
trueSquareSlopeInhale = +0.1 # x-axis
trueSquareSlopeExhale = -0.06 # x-axis
numFigures = 0 
if (motion == 'Step'): nFrames = 2 
else: nFrames = 36
noiseLevel = 10 

# Store all settings in a text file 
mf.write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueSlope, trueSlopeInhale, trueSlopeExhale, trueSquareSlopeInhale, trueSquareSlopeExhale, nFrames)


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
phantomList, surSignal, shiftList, shiftXList = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueSlope, trueSlopeInhale, trueSlopeExhale, trueSquareSlopeInhale, trueSquareSlopeExhale, image2D, stationary)
originalImage = phantomList[0]

# Plot hysteresis on x-axis
plt.figure() 
plt.plot(surSignal, shiftXList), plt.title('Hysteresis (x-axis)'), plt.xlabel('Surrogate signal (external motion)'), plt.ylabel('Internal motion x-axis')
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_Hysteresis.png'.format(numFigures, trueShiftAmplitude)), plt.close()
plt.show()
numFigures += 1 

# Plot hysteresis on y-axis
plt.figure() 
plt.plot(surSignal, shiftList), plt.title('Hysteresis (y-axis)'), plt.xlabel('Surrogate signal (external motion)'), plt.ylabel('Internal motion y-axis')
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_Hysteresis.png'.format(numFigures, trueShiftAmplitude)), plt.close()
plt.show()
numFigures += 1 

# Plot a time series of the phantom 
'''
for iFrame in range(nFrames):    
    plt.title('Time frame {0}'.format(iFrame)), plt.imshow(phantomList[iFrame][0,:,:], interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r)
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom_TF{}.png'.format(numFigures, trueShiftAmplitude, iFrame)), plt.close()
    numFigures += 1 
'''


#_________________________DISTINGUISH INHALE AND EXHALE PHASES_______________________________ 
# Derivatives, the sign of which distinguishes between inhale and exhale 
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
plt.subplot(1,2,1), plt.title('Without noise'), plt.imshow(measNoNoise, interpolation=None, vmin = 0, vmax = noiseLevel, cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('With noise'), plt.imshow(measWithNoise, interpolation=None, vmin = 0, vmax = noiseLevel, cmap=plt.cm.Greys_r)
plt.suptitle('Time Frame 1'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_measurementsWithWithoutNoise.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

# Initial guess - image 
guess = np.ones(np.shape(image2D))
# Initial guess - model 
slopeFound = 0.0 

# Plot and save initial guess 
plt.figure(), plt.title('Initial guess'), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(guess), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 
guessTMP = np.zeros((1,) + np.shape(image2D))
guessTMP[0,:,:] = guess
pyvpx.numpy2vpx(guessTMP, figSaveDir + 'guess.vpx') 


#_________________________NESTED EM LOOP_______________________________
# Lists for storage 
slopeFoundList = []
quadErrorSumFoundList = []
quadErrorSumListList = []
guessSum = []
guessSum.append(np.sum(guess))

for iIt in range(nIt): 
    # Motion model optimization
    if (iIt >= 4):
        quadErrorSumList = []   
        
        # For each slope in slopeList, compute the quadratic error 
        slopeList = np.linspace(trueSlope-1., trueSlope+1., 9)
        for slope in slopeList: 
            quadErrorSum = 0 
            for iFrame in range(nFrames): 
                guessMoved = np.zeros(np.shape(guess))
                guessMoved = sp.ndimage.shift(copy.deepcopy(guess), (surSignal[iFrame] * slope, 0)) 
                guessMovedProj = radon(copy.deepcopy(guessMoved), iAngles)
                quadErrorSum += np.sum((guessMovedProj - measList[iFrame])**2)
            
            quadErrorSumList.append({'slope' : slope, 'quadErrorSum' : quadErrorSum})
            print 'Slope: {}'.format(slope), 'Quadratic error: {}'.format(quadErrorSum)

        # Find the slope in slopeList that gives the minimum quadratic error 
        quadErrorSums = [x['quadErrorSum'] for x in quadErrorSumList]
        index = quadErrorSums.index(np.min(quadErrorSums))
        slopeFound = quadErrorSumList[index]['slope']
        quadErrorSumFound = quadErrorSumList[index]['quadErrorSum']

        # Store stuff 
        quadErrorSumListList.append(quadErrorSums)
        slopeFoundList.append(slopeFound)
        quadErrorSumFoundList.append(quadErrorSumFound) 

        # Fit quadratic function to the quadratic error 
        '''
        def func(x, a, b, c): 
            return a * (x-b)**2 + c
        popt, pcov = curve_fit(func, slopeList, quadErrorSums)
        plt.plot(slopeList, func(slopeList, *popt), 'g-', label = 'fit')
        '''

        # Plot 
        plt.plot(slopeList, quadErrorSums, 'b-', slopeFound, quadErrorSumFound, 'ro'), plt.title('Quadratic error vs. slope')
        plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError_Iteration{}.png'.format(numFigures, trueShiftAmplitude, iIt))
        numFigures += 1 
        plt.close()

    totalError = 0 
    # MLEM with motion compensation 
    for iFrame in range(nFrames): 
        # Shift guess for the current model, in time frame iFrame, and forward project it 
        shiftedGuess = np.zeros(np.shape(guess)) 
        shiftedGuess = sp.ndimage.shift(copy.deepcopy(guess), (surSignal[iFrame] * slopeFound, 0)) 
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
for i in range(len(quadErrorSumListList)): 
    plt.plot(slopeFoundList, quadErrorSumFoundList, 'ro') 
    plt.plot(slopeList, quadErrorSumListList[i], label = 'Iteration {}'.format(i+1)), plt.title('Quadratic error vs. slope')
    plt.axvline(trueSlope, color='k', linestyle='--')
plt.legend()
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close()