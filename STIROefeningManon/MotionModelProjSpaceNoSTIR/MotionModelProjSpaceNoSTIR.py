import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon
import ManonsFunctions as mf 
import scipy as sp

#phantom = 'Block'
phantom = 'Shepp-Logan' 
#noise = False
noise = True
#motion = 'Step' 
motion = 'Sine'
#stationary = True 
stationary = False # Only possible for sinusoidal motion 

#gating = True
gating = False 
nIt = 10 
trueShiftAmplitude = 15 # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd) + LET OP: kan niet groter zijn dan de lengte van het plaatje (kan de code niet aan) 
trueOffset = 5
numFigures = 0 
duration = 60 # in seconds
if (motion == 'Step'): nFrames = 2
else: nFrames = 80

figSaveDir = mf.make_figSaveDir(motion, phantom, noise, stationary, gating)

mf.write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueOffset, duration, nFrames, gating)

#_________________________MAKE PHANTOM_______________________________
image2D = mf.make_Phantom(phantom, duration)
plt.figure(), plt.title('Original image'), plt.imshow(image2D, interpolation = None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1
 
#_________________________ADD MOTION_______________________________ 
phantomList, surSignal, shiftList, gateMin, gateMax = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueOffset, image2D, stationary, gating)
originalImage = phantomList[0]

x = np.arange(0, nFrames, 0.1)
plt.plot(range(nFrames), surSignal, 'bo', label = 'Surrogate signal', markersize = 3), plt.title('Sinusoidal phantom shifts'), plt.xlabel('Time frame'), plt.ylabel('Shift')
plt.plot(range(len(shiftList)), shiftList, 'ro', label = 'True motion', markersize = 3) 
plt.axhline(y = gateMin, color = 'grey', label = 'Respiratory gating')
plt.axhline(y = gateMax, color = 'grey')
plt.fill_between(x, gateMin, gateMax, color='grey', alpha='0.5')
plt.legend(loc = 0), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_shiftList.png'.format(numFigures, trueShiftAmplitude)), plt.close()
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
    measList.append(meas) 

plt.subplot(1,2,1), plt.title('Without noise'), plt.imshow(measNoNoise, interpolation=None, vmin = 0, vmax = 1000)
plt.subplot(1,2,2), plt.title('With noise'), plt.imshow(measWithNoise, interpolation=None, vmin = 0, vmax = 1000)
plt.suptitle('Time Frame 1'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_measurementsWithWithoutNoise.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

reconList = []
for iFrame in range(len(measList)): 
    reconList.append(iradon(measList[iFrame], iAngles)) 
guess = np.mean(reconList, axis = 0)
plt.figure(), plt.title('Initial guess'), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

normSino = np.ones(np.shape(measList[0]))
norm = iradon(normSino, iAngles, filter = None) # We willen nu geen ramp filter

#_________________________NESTED EM LOOP_______________________________
offsetFoundList = []
quadErrorSumFoundList = []
quadErrorSumListList = []
offsetFound = trueOffset-1 # First guess 
guessSum = []
guessSum.append(np.sum(guess))
for iIt in range(nIt): 
    # Normal MLEM 
    for iFrame in range(nFrames): 
        shiftedGuess = np.zeros(np.shape(guess))
        sp.ndimage.shift(guess, (surSignal[iFrame] - offsetFound, 0), shiftedGuess)
        shiftedGuessSinogram = radon(shiftedGuess, iAngles) 
        error = measList[iFrame]/shiftedGuessSinogram 
        error[np.isnan(error)] = 0
        error[np.isinf(error)] = 0
        error[error > 1E10] = 0;
        error[error < 1E-10] = 0
        errorBck = iradon(error, iAngles, filter = None) 
        errorBckShifted = np.zeros(np.shape(errorBck))
        sp.ndimage.shift(errorBck, (-surSignal[iFrame] + offsetFound, 0), errorBckShifted)
        guess *= errorBckShifted
    guess /= norm 
    guessSum.append(np.sum(guess))
    countIt = iIt+1 

    # Motion model optimization
    guessMovedList = []
    guessMovedProjList = []
    quadErrorSumList = []   
    offsetList = range(trueOffset-4, trueOffset+5)
    for offset in offsetList: 
        quadErrorSum = 0 
        for iFrame in range(nFrames): 
            guessMovedList.append(np.zeros(np.shape(guess)))
            sp.ndimage.shift(guess, (surSignal[iFrame] - offset, 0), guessMovedList[iFrame]) # Je bent als het ware de correctie op het surrogaat signaal aan het zoeken
            guessMovedProj = radon(guessMovedList[iFrame], iAngles)
            guessMovedProjList.append(guessMovedProj) 
            quadErrorSum += np.sum((guessMovedProj - measList[iFrame])**2)
        quadErrorSumList.append({'offset' : offset, 'quadErrorSum' : quadErrorSum})

    quadErrorSums = [x['quadErrorSum'] for x in quadErrorSumList]
    for i in range(len(quadErrorSumList)): 
        if(quadErrorSumList[i]['quadErrorSum'] == min(quadErrorSums)): 
            offsetFound = quadErrorSumList[i]['offset']
            offsetFoundList.append(offsetFound)
            quadErrorSumFound = quadErrorSumList[i]['quadErrorSum']
            quadErrorSumFoundList.append(quadErrorSumFound) 

    quadErrorSumListList.append(quadErrorSums)

    #plt.plot(offsetList, quadErrorSums, 'b-', offsetFound, quadErrorSumFound, 'ro'), plt.title('Quadratic error vs. offset TEST')
    #plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError_Iteration{}.png'.format(numFigures, trueShiftAmplitude, iIt))
    #numFigures += 1 
    #plt.close()

    # Motion compensation 
    reconCorList = [] 
    for iFrame in range(nFrames):
        reconCorList.append(np.zeros(np.shape(guess))) 
        sp.ndimage.shift(guess, (0, 0), reconCorList[iFrame]) 
    
    guess = np.mean(reconCorList, axis = 0)
    guessSum.append(np.sum(guess))

    plt.figure(), plt.title('Guess after {0} iteration(s)'.format(iIt+1)), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_finalImage.png'.format(numFigures, trueShiftAmplitude)), plt.close()
    numFigures += 1  

plt.figure(), plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(originalImage[0,:,:], interpolation=None, vmin = 0, vmax = np.max(image2D))
plt.subplot(1,2,2), plt.title('Reconstructed Image'), plt.imshow(guess, interpolation=None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_originalAndRecon.png'.format(numFigures, trueShiftAmplitude)), plt.close() 
numFigures += 1 

for i in range(len(quadErrorSumListList)): 
    if(i%(nIt/5) == 0):
        plt.plot(offsetFoundList, quadErrorSumFoundList, 'ro') 
        plt.plot(offsetList, quadErrorSumListList[i], label = 'Iteration {}'.format(i)), plt.title('Quadratic error vs. offset')
        plt.axvline(trueOffset, color='k', linestyle='--')
plt.legend()
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close()