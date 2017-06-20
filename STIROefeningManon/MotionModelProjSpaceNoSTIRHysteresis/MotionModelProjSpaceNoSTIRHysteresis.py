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
from matplotlib.ticker import MaxNLocator

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
modelBroken = True  

# Create a direcotory for figure storage (just the string, make sure  the folder already exists!) 
dir = './Figures/'
figSaveDir = mf.make_figSaveDir(dir, motion, phantom, noise, stationary, modelBroken)

# Parameters that do not influence the saving directory 
nIt = 6
nModelSkip = 3
trueShiftAmplitude = 10 # Make sure this is not too large, activity moving out of the FOV will cause problems 
trueSlope = 0.5 # y-axis 
trueSlopeX = 0.0 # x-axis 
trueSlopeInhale = 1.0 # hysteresis, x-axis
trueSlopeExhale = trueSlopeInhale # hysteresis, x-axis, must be the same as trueSlopeInhale, otherwise the two functions do are not equal at the endpoints
trueSquareSlopeInhale = +0.1 # hysteresis, x-axis
trueSquareSlopeExhale = -0.06 # hysteresis, x-axis
numFigures = 0 
if (motion == 'Step'): nFrames = 2 
else: nFrames = 18
noiseLevel = 600
x0 = np.array([1.0, 1.0]) # initial guess for the optimization function 

# Store all settings in a text file 
mf.write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueSlope, trueSlopeInhale, trueSlopeExhale, trueSquareSlopeInhale, trueSquareSlopeExhale, nFrames, hysteresis, x0, modelBroken)


#_________________________MAKE PHANTOM_______________________________
# Make phantom 
image2D = mf.make_Phantom(phantom, noiseLevel)

# Plot phantom and store as vpx image 
plt.figure(), plt.title('Original image'), plt.imshow(image2D, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_phantom.png'.format(numFigures)), plt.close()
numFigures += 1

image2DTMP = np.zeros((1,) + np.shape(image2D) )
image2DTMP[0,:,:] = image2D
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'OriginalImage.vpx') 
 

#_________________________ADD MOTION_______________________________ 
# Create surrogate signal and add motion to the phantom  
phantomList, surSignal, shiftList, shiftXList = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueSlope, trueSlopeX, trueSlopeInhale, trueSlopeExhale, trueSquareSlopeInhale, trueSquareSlopeExhale, image2D, stationary, hysteresis, modelBroken)
originalImage = phantomList[0]

'''
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
plt.plot(range(nFrames), shiftList, label = 'True motion y-axis'), plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_shiftList.png'.format(numFigures)), plt.close()
numFigures += 1 

if (hysteresis): 
    # x-axis, inhale
    plt.figure() 
    plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Motion (y-axis, inhale)'), plt.xlabel('Time frame'), plt.ylabel('Shift')
    plt.plot(inhaleSurAxis, inhaleSurSignal, 'ro', label = 'Inhale surrogate') 
    plt.plot(range(nFrames), shiftXList, label = 'Internal motion')
    plt.plot(inhaleShiftXAxis, inhaleShiftXList, 'ro') 
    plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_Inhale.png'.format(numFigures)), plt.close()
    numFigures += 1 
    # x-axis, exhale
    plt.figure() 
    plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Motion (y-axis, exhale)'), plt.xlabel('Time frame'), plt.ylabel('Shift')
    plt.plot(exhaleSurAxis, exhaleSurSignal, 'go', label = 'Exhale') 
    plt.plot(range(nFrames), shiftXList, label = 'Internal motion')
    plt.plot(exhaleShiftXAxis, exhaleShiftXList, 'go') 
    plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_Exhale.png'.format(numFigures)), plt.close()
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
    measList.append(meas.astype(np.float)) 

# Plot sinogram of time frame 0 with and without noise  
plt.figure() 
plt.subplot(1,2,1), plt.title('Without noise'), plt.imshow(measNoNoise, interpolation=None, vmin = 0, vmax = np.max(measWithNoise), cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('With noise'), plt.imshow(measWithNoise, interpolation=None, vmin = 0, vmax = np.max(measWithNoise), cmap=plt.cm.Greys_r)
plt.suptitle('Time Frame 1'), plt.savefig(figSaveDir + 'Fig{}_measurementsWithWithoutNoise.png'.format(numFigures)), plt.close()
numFigures += 1 

# Initial guess - image 
guess = np.ones(np.shape(image2D)) # Fills it with floats, not ints 

# Plot and save initial guess 
#plt.figure(), plt.title('Initial guess'), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(guess), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude)), plt.close()
#numFigures += 1 
guessTMP = np.zeros((1,) + np.shape(image2D))
guessTMP[0,:,:] = guess
pyvpx.numpy2vpx(guessTMP, figSaveDir + 'guess.vpx') 
    
# Objective function for model optimization
def computeQuadError(x, nFrames, guess, surSignal, iAngles, returnTuple):    
    quadErrorSum = 0.0
    quadErrorSumList = []
    
    for iFrame in range(nFrames): 
        guessMoved = np.zeros(np.shape(guess))
        guessMoved = sp.ndimage.shift(copy.deepcopy(guess), (surSignal[iFrame] * x[0], 0)) # shift in y-direction 
        guessMoved = sp.ndimage.shift(copy.deepcopy(guessMoved), (0, surSignal[iFrame] * x[1])) # shift in x-direction 
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

slopeList = np.linspace(-1, 2, 9)

parFile = open(figSaveDir + "Parameters.txt", "w")

# Lists for storage 
quadErrorsList = []
slopeFoundList = []
slopeXFoundList = []
quadErrorFoundList = []
guessSum = []
guessSum.append(np.sum(guess))
for iIt in range(nIt): 
    # Motion model optimization
    if (iIt >= nModelSkip):
        args = (nFrames, guess, surSignal, iAngles, False)
        myOptions = {'disp': True, 'maxiter' : 10}
        if (iIt <= (nIt-4)): # The first iterations are less accurate in terms of tolerance value and step size 
            myOptions['gtol'] = 1e-02
            myOptions['eps'] = 1e-04
        res = minimize(computeQuadError, x0, args, method = 'BFGS', options = myOptions)
        slopeFound = res.x[0]        
        slopeFoundList.append(slopeFound)
        slopeXFound = res.x[1]  
        slopeXFoundList.append(slopeXFound) 
        quadErrorFound = res.fun
        quadErrorFoundList.append(quadErrorFound)  

        # Rough estimate of the variation in quadratic error as a function of the slope (of the y-axis) 
        #quadErrors = [computeQuadError(np.array([i, slopeXFound]), nFrames, guess, surSignal, iAngles, False) for i in slopeList]
        #quadErrorsList.append(quadErrors)

        # Time-resolved quadratic error 
        #quadErrorSumList = computeQuadError((slopeFound, slopeXFound), nFrames, guess, surSignal, iAngles, True)   
        quadErrorSumList = computeQuadError((trueSlope, trueSlopeX), nFrames, guess, surSignal, iAngles, True)   
    
        # Moving average window for time-resolved quadratic error 
        #windowLength = 10 
        #window = np.ones(windowLength,'d')
        #quadErrorSumListAVG = np.convolve(window/window.sum(), quadErrorSumList, mode='same')

        plt.figure() 
        plt.plot(quadErrorSumList), plt.title('Quadratic error vs. time, iteration {}'.format(iIt+1))
        plt.axis([0.0, nFrames, 0.0, 1.0])
        plt.savefig(figSaveDir + 'Fig{}_QuadraticError_Time.png'.format(numFigures))
        numFigures += 1 
        plt.close() 

        print 'Slope found: {}'.format(slopeFound)
        print 'SlopeX found: {}'.format(slopeXFound)
        parFile.write('Iteration {}\n'.format(iIt+1))
        parFile.write('objective function: {}\n'.format(res.fun))
        parFile.write('slope: {}\n'.format(slopeFound)) 
        parFile.write('slopeX: {}\n\n'.format(slopeXFound)) 

        '''
        plt.plot(slopeList, quadErrors, 'b-', label = ''), plt.title('Quadratic error vs. slope, iteration {}'.format(iIt+1))
        plt.plot(slopeFound, quadErrorFound, 'ro', label = 'Estimated value')
        plt.axvline(trueSlope, color='k', linestyle='--', label = 'Correct  value')
        plt.legend()
        plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError_Iteration{}.png'.format(numFigures, trueShiftAmplitude, iIt))
        numFigures += 1 
        plt.close()
        ''' 

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
    #guessTMP = np.zeros((1,) + np.shape(image2D))
    #guessTMP[0,:,:] = guess
    #pyvpx.numpy2vpx(guessTMP, figSaveDir + 'guess_{}.vpx'.format(iIt)) 
    plt.figure(), plt.title('Guess after {} iteration(s)'.format(iIt+1)), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_finalImage.png'.format(numFigures)), plt.close()
    numFigures += 1  

parFile.close() 

# Plot original image and reconstructed image 
plt.figure(), plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(originalImage[0,:,:], interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('Reconstructed Image'), plt.imshow(guess, interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_originalAndRecon.png'.format(numFigures)), plt.close() 
numFigures += 1 

# Plot some of guess as a function of iteration number 
'''
plt.figure() 
plt.plot(guessSum), plt.title('Sum of guess'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_guessSum.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close() 
'''

# Plot quadratic errors of all iteqrations
# y-axis
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # to get integer values on the x-axis
plt.axhline(trueSlope, color = 'k', linestyle = '--', label = 'Correct value')
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

qualityFile = open(figSaveDir + "Quality.txt", "w")
qualityFile.write('Phantom:\n')
qualityFile.write('Maximum value: {}\n\n'.format(np.max(image2D))) 
qualityFile.write('Simulations:\n')
qualityFile.write('Maximum value: {}\n\n'.format(np.max(guess))) 
qualityFile.write('Quadratic difference of simulation and phantom: {}\n'.format(np.sum( (guess - image2D)**2 )))
qualityFile.close() 