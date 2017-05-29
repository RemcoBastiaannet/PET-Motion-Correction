import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon
import ManonsFunctionsHysteresis as mf 
import scipy as sp
import pyvpx
import copy
from scipy.optimize import curve_fit

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
trueShiftAmplitude = 5 # Make sure this is not too large, activity moving out of the FOV will cause problems 
trueSlope = 0.5 
trueSquareSlope = 0.04 
numFigures = 0 
if (motion == 'Step'): nFrames = 2 
else: nFrames = 36
noiseLevel = 10 

# Store all settings in a text file 
mf.write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueSlope, trueSquareSlope, nFrames)

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
phantomList, surSignal, shiftList, shiftXList = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueSlope, trueSquareSlope, image2D, stationary)
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

# Plot surrogate signal and internal motion 
# x-axis 
plt.figure()
plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Motion (x-axis)'), plt.xlabel('Time frame'), plt.ylabel('Shift')
plt.plot(range(nFrames), shiftXList, label = 'True motion x-axis'), plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_shiftXList.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

# y-axis
plt.figure()
plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Motion (y-axis)'), plt.xlabel('Time frame'), plt.ylabel('Shift')
plt.plot(range(nFrames), shiftList, label = 'True motion y-axis'), plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_shiftList.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

#_________________________MEASUREMENT, INITIAL GUESS, NORMALIZATION_______________________________
iAngles = np.linspace(0, 360, 120)[:-1]

measList = []
for iFrame in range(nFrames):
    meas = radon(copy.deepcopy(phantomList[iFrame])[0,:,:], iAngles) 
    if (iFrame == 0): measNoNoise = meas
    if (noise): meas = sp.random.poisson(meas)
    if (iFrame == 0): measWithNoise = meas
    measList.append(meas) 

plt.figure() 
plt.subplot(1,2,1), plt.title('Without noise'), plt.imshow(measNoNoise, interpolation=None, vmin = 0, vmax = noiseLevel, cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('With noise'), plt.imshow(measWithNoise, interpolation=None, vmin = 0, vmax = noiseLevel, cmap=plt.cm.Greys_r)
plt.suptitle('Time Frame 1'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_measurementsWithWithoutNoise.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

guess = np.ones(np.shape(image2D))
plt.figure(), plt.title('Initial guess'), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(guess), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 
guessTMP = np.zeros((1,) + np.shape(image2D))
guessTMP[0,:,:] = guess
pyvpx.numpy2vpx(guessTMP, figSaveDir + 'guess.vpx') 

#_________________________NESTED EM LOOP_______________________________
slopeFoundList = []
quadErrorSumFoundList = []
quadErrorSumListList = []
guessSum = []
guessSum.append(np.sum(guess))
slopeFound = 0.0 # Initial guess  
for iIt in range(nIt): 
    if (iIt >= 4):
        # Motion model optimization
        quadErrorSumList = []   
        slopeList = np.linspace(trueSlope-1., trueSlope+1., 9)
        for slope in slopeList: 
            quadErrorSum = 0 
            for iFrame in range(nFrames): 
                guessMoved = np.zeros(np.shape(guess))
                guessMoved = sp.ndimage.shift(copy.deepcopy(guess), (surSignal[iFrame] * slope, 0)) # Je bent als het ware de correctie op het surrogaat signaal aan het zoeken
                guessMovedProj = radon(copy.deepcopy(guessMoved), iAngles)
                quadErrorSum += np.sum(abs(guessMovedProj - measList[iFrame]))
            quadErrorSumList.append({'slope' : slope, 'quadErrorSum' : quadErrorSum})
            print 'Slope: {}'.format(slope), 'Quadratic error: {}'.format(quadErrorSum)

        quadErrorSums = [x['quadErrorSum'] for x in quadErrorSumList]
        quadErrorSumListList.append(quadErrorSums)
        index = quadErrorSums.index(np.min(quadErrorSums))
        slopeFound = quadErrorSumList[index]['slope']
        slopeFoundList.append(slopeFound)
        quadErrorSumFound = quadErrorSumList[index]['quadErrorSum']
        quadErrorSumFoundList.append(quadErrorSumFound) 

        '''
        def func(x, a, b, c): 
            return a * (x-b)**2 + c
        popt, pcov = curve_fit(func, slopeList, quadErrorSums)
        plt.plot(slopeList, func(slopeList, *popt), 'g-', label = 'fit')
        '''

        plt.plot(slopeList, quadErrorSums, 'b-', slopeFound, quadErrorSumFound, 'ro'), plt.title('Quadratic error vs. slope')
        plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError_Iteration{}.png'.format(numFigures, trueShiftAmplitude, iIt))
        numFigures += 1 
        plt.close()

    totalError = 0 
    # MLEM with motion compensation 
    for iFrame in range(nFrames): 
        shiftedGuess = np.zeros(np.shape(guess)) 
        shiftedGuess = sp.ndimage.shift(copy.deepcopy(guess), (surSignal[iFrame] * slopeFound, 0)) 
        shiftedGuessSinogram = radon(shiftedGuess, iAngles) 
        error = measList[iFrame]/shiftedGuessSinogram 
        error[np.isnan(error)] = 0
        error[np.isinf(error)] = 0
        error[error > 1E10] = 0
        error[error < 1E-10] = 0
        errorBck = iradon(error, iAngles, filter = None) 
        errorBckShifted = np.zeros(np.shape(errorBck)) 
        errorBckShifted = sp.ndimage.shift(errorBck, (-surSignal[iFrame] * slopeFound, 0)) 
        totalError += errorBckShifted   
    guess *= totalError/nFrames
    guess /= np.sum(guess) 
    guess *= np.sum(measList[-1])/np.shape(measList[-1])[1] 
    guessTMP = np.zeros((1,) + np.shape(image2D))
    guessTMP[0,:,:] = guess
    pyvpx.numpy2vpx(guessTMP, figSaveDir + 'guess_{}.vpx'.format(iIt)) 
    guessSum.append(np.sum(guess))
    countIt = iIt+1 

    plt.figure(), plt.title('Guess after {} iteration(s)'.format(iIt+1)), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_finalImage.png'.format(numFigures, trueShiftAmplitude)), plt.close()
    numFigures += 1  

plt.figure(), plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(originalImage[0,:,:], interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('Reconstructed Image'), plt.imshow(guess, interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_originalAndRecon.png'.format(numFigures, trueShiftAmplitude)), plt.close() 
numFigures += 1 

plt.figure() 
plt.plot(guessSum), plt.title('Sum of guess'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_guessSum.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close() 

for i in range(len(quadErrorSumListList)): 
    plt.plot(slopeFoundList, quadErrorSumFoundList, 'ro') 
    plt.plot(slopeList, quadErrorSumListList[i], label = 'Iteration {}'.format(i+1)), plt.title('Quadratic error vs. slope')
    plt.axvline(trueSlope, color='k', linestyle='--')
plt.legend()
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close()