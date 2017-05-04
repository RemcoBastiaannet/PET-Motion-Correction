import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon
import ManonsFunctions as mf 
import scipy as sp
import pyvpx

#phantom = 'Block'
phantom = 'Shepp-Logan' 
#noise = False
noise = True
#motion = 'Step' 
motion = 'Sine'
#stationary = True 
stationary = False # Only possible for sinusoidal motion 

nIt = 10
trueShiftAmplitude = 15 # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd) + LET OP: kan niet groter zijn dan de lengte van het plaatje (kan de code niet aan) 
trueOffset = 5
numFigures = 0 
duration = 60 # in seconds
if (motion == 'Step'): nFrames = 2
else: nFrames = 20
noiseLevel = 10 
gating = False 

dir = './Figures/'
figSaveDir = mf.make_figSaveDir(dir, motion, phantom, noise, stationary)

mf.write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueOffset, duration, nFrames, gating)

#_________________________MAKE PHANTOM_______________________________
image2D = mf.make_Phantom(phantom, duration, noiseLevel)
plt.figure(), plt.title('Original image'), plt.imshow(image2D, interpolation = None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1

image2DTMP = np.zeros((1, 460, 460))
image2DTMP[0,:,:] = image2D
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'image2D.vpx') 
 
#_________________________ADD MOTION_______________________________ 
phantomList, surSignal, shiftList = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueOffset, image2D, stationary)
originalImage = phantomList[0]

for iFrame in range(nFrames):    
    plt.subplot(2,nFrames/2+1,iFrame+1), plt.title('Time frame {0}'.format(iFrame)), plt.imshow(phantomList[iFrame][0,:,:], interpolation=None, vmin = 0, vmax = np.max(image2D)) 
plt.suptitle('Phantom'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 
 
plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Sinusoidal phantom shifts'), plt.xlabel('Time frame'), plt.ylabel('Shift')
plt.plot(range(nFrames), shiftList, label = 'True motion'), plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_shiftList.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

#_________________________MEASUREMENT, INITIAL GUESS, NORMALIZATION_______________________________
iAngles = np.linspace(0, 360, 120)[:-1]

measList = []
for iFrame in range(nFrames):
    meas = radon(phantomList[iFrame][0,:,:], iAngles) 
    if (iFrame == 0): measNoNoise = meas
    if (noise): 
        meas = sp.random.poisson(meas)
    if (iFrame == 0): measWithNoise = meas
    plt.subplot(2,nFrames/2+1,iFrame+1), plt.title('Time frame {0}'.format(iFrame)), plt.imshow(meas, interpolation=None, vmin = 0, vmax = 1000) 
    measList.append(meas) 
plt.suptitle('Measurements'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_measurements.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

measListTMP = np.zeros((1, 651, 119))
measListTMP[0,:,:] = measList[0]
pyvpx.numpy2vpx(measListTMP, figSaveDir + 'measurementTF1.vpx')


plt.subplot(1,2,1), plt.title('Without noise'), plt.imshow(measNoNoise, interpolation=None, vmin = 0, vmax = noiseLevel)
plt.subplot(1,2,2), plt.title('With noise'), plt.imshow(measWithNoise, interpolation=None, vmin = 0, vmax = noiseLevel)
plt.suptitle('Time Frame 1'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_measurementsWithWithoutNoise.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

reconList = []
for iFrame in range(len(measList)): 
    reconList.append(iradon(measList[iFrame], iAngles)) 
guess = np.mean(reconList, axis = 0)
#guess = reconList[0] # Added
plt.figure(), plt.title('Initial guess'), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

guessTMP = np.zeros((1, 460, 460))
guessTMP[0,:,:] = guess
pyvpx.numpy2vpx(guessTMP, (figSaveDir + 'guess.vpx'))

normSino = np.ones(np.shape(measList[0]))
norm = iradon(normSino, iAngles, filter = None) # We willen nu geen ramp filter
plt.figure(), plt.title('MLEM normalization'), plt.imshow(norm, interpolation = None, vmin = 0, vmax = 0.03), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_norm.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1  

#_________________________NESTED EM LOOP_______________________________
offsetFoundList = []
quadErrorSumFoundList = []
quadErrorSumListList = []
#offsetFound = trueOffset-2 # First guess 
guessSum = []
guessSum.append(np.sum(guess))
for iIt in range(nIt): 
    # Motion model optimization
    guessMovedList = []
    guessMovedProjList = []
    quadErrorSumList = []   
    offsetList = range(trueOffset-2, trueOffset+3)
    for offset in offsetList: 
        quadErrorSum = 0 
        for iFrame in range(nFrames): 
            guessMovedList.append(np.zeros(np.shape(guess))) 
            sp.ndimage.shift(guess, (surSignal[iFrame] - offset, 0), guessMovedList[iFrame]) # Je bent als het ware de correctie op het surrogaat signaal aan het zoeken
            guessMovedProj = radon(guessMovedList[iFrame], iAngles)
            guessMovedProjList.append(guessMovedProj) 
            quadErrorSum += np.sum((guessMovedProj - measList[iFrame])**2)
        quadErrorSumList.append({'offset' : offset, 'quadErrorSum' : quadErrorSum})
        print 'Offset: {}'.format(offset), 'Quadratic error: {}'.format(quadErrorSum)

    quadErrorSums = [x['quadErrorSum'] for x in quadErrorSumList]
    for i in range(len(quadErrorSumList)): 
        if(quadErrorSumList[i]['quadErrorSum'] == min(quadErrorSums)): 
            offsetFound = quadErrorSumList[i]['offset']
            offsetFoundList.append(offsetFound)
            quadErrorSumFound = quadErrorSumList[i]['quadErrorSum']
            quadErrorSumFoundList.append(quadErrorSumFound) 
    quadErrorSumListList.append(quadErrorSums)

    plt.plot(offsetList, quadErrorSums, 'b-', offsetFound, quadErrorSumFound, 'ro'), plt.title('Quadratic error vs. offset TEST')
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError_Iteration{}.png'.format(numFigures, trueShiftAmplitude, iIt))
    numFigures += 1 
    plt.close()

    # Normal MLEM 
    for iFrame in range(nFrames): 
        shiftedGuess = np.zeros(np.shape(guess))
        #shiftedGuess = guess
        sp.ndimage.shift(guess, (surSignal[iFrame] - offsetFound, 0), shiftedGuess)
        shiftedGuessTMP = np.zeros((1, 460, 460))
        shiftedGuessTMP[0,:,:] = shiftedGuess
        pyvpx.numpy2vpx(shiftedGuessTMP, figSaveDir + 'shiftedGuessIteration{}.vpx'.format(iIt))
        shiftedGuessSinogram = radon(shiftedGuess, iAngles) 
        shiftedGuessSinogramTMP = np.zeros((1, 651, 119))
        shiftedGuessSinogramTMP[0,:,:] = shiftedGuessSinogram 
        pyvpx.numpy2vpx(shiftedGuessSinogramTMP, (figSaveDir + 'shiftedGuessSinogramIteration{}.vpx'.format(iIt)))
        error = measList[iFrame]/shiftedGuessSinogram 
        errorTMP = np.zeros((1, 651, 119))
        errorTMP[0,:,:] = error 
        pyvpx.numpy2vpx(errorTMP, figSaveDir + 'errorIteration{}.vpx'.format(iIt))
        error[np.isnan(error)] = 0
        error[np.isinf(error)] = 0
        error[error > 1E10] = 0
        error[error < 1E-10] = 0
        errorBck = iradon(error, iAngles, filter = None) 
        errorBckTMP = np.zeros((1, 460, 460))
        errorBckTMP[0,:,:] = errorBck 
        pyvpx.numpy2vpx(errorBckTMP, figSaveDir + 'errorBck{}.vpx'.format(iIt))
        errorBckShifted = np.zeros(np.shape(errorBck))
        sp.ndimage.shift(errorBck, (-surSignal[iFrame] + offsetFound, 0), errorBckShifted)
        #errorBckShifted = errorBck
        guess *= errorBckShifted
    guess /= norm 
    guessSum.append(np.sum(guess))
    countIt = iIt+1 

    plt.figure(), plt.title('Guess after {0} iteration(s)'.format(iIt+1)), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_finalImage.png'.format(numFigures, trueShiftAmplitude)), plt.close()
    numFigures += 1  

plt.figure(), plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(originalImage[0,:,:], interpolation=None, vmin = 0, vmax = np.max(image2D))
plt.subplot(1,2,2), plt.title('Reconstructed Image'), plt.imshow(guess, interpolation=None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_originalAndRecon.png'.format(numFigures, trueShiftAmplitude)), plt.close() 
numFigures += 1 

plt.plot(guessSum), plt.title('Sum of guess'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_guessSum.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close() 

for i in range(len(quadErrorSumListList)): 
    plt.plot(offsetFoundList, quadErrorSumFoundList, 'ro') 
    plt.plot(offsetList, quadErrorSumListList[i], label = 'Iteration {}'.format(i)), plt.title('Quadratic error vs. offset')
    plt.axvline(trueOffset, color='k', linestyle='--')
plt.legend()
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close()