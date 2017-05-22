import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon
import ManonsFunctions as mf 
import scipy as sp
import pyvpx
import copy

#phantom = 'Block'
phantom = 'Shepp-Logan' 
noise = False
#noise = True
#motion = 'Step' 
motion = 'Sine'
stationary = True 
#stationary = False # Only possible for sinusoidal motion 

nIt = 10
trueShiftAmplitude = 13 # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd) + LET OP: kan niet groter zijn dan de lengte van het plaatje (kan de code niet aan) 
trueOffset = 5
numFigures = 0 
duration = 60 # in seconds
if (motion == 'Step'): nFrames = 2
else: nFrames = 3 
noiseLevel = 10 
gating = False 

dir = './Figures/'
figSaveDir = mf.make_figSaveDir(dir, motion, phantom, noise, stationary)

mf.write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueOffset, duration, nFrames, gating)

#_________________________MAKE PHANTOM_______________________________
image2D = mf.make_Phantom(phantom, duration, noiseLevel)
'''
Lx = Ly = 50
image2D = np.zeros((Lx,Ly))
for x in range(Lx): 
    for y in range(Ly): 
        if((x-Lx/2)**2 + (y-Ly/2)**2 <= 30): image2D[x,y] = 1
'''
plt.figure(), plt.title('Original image'), plt.imshow(image2D, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1
image2DTMP = np.zeros((1,) + np.shape(image2D) )
image2DTMP[0,:,:] = image2D
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'image2D.vpx') 
 
#_________________________ADD MOTION_______________________________ 
phantomList, surSignal, shiftList = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueOffset, image2D, stationary)
originalImage = phantomList[0]

for iFrame in range(nFrames):    
    plt.subplot(2,nFrames/2+1,iFrame+1), plt.title('Time frame {0}'.format(iFrame)), plt.imshow(phantomList[iFrame][0,:,:], interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r) 
plt.suptitle('Phantom'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 
 
plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Sinusoidal phantom shifts'), plt.xlabel('Time frame'), plt.ylabel('Shift')
plt.plot(range(nFrames), shiftList, label = 'True motion'), plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_shiftList.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

#_________________________MEASUREMENT, INITIAL GUESS, NORMALIZATION_______________________________
iAngles = np.linspace(0, 360, 120)[:-1]

measList = []
for iFrame in range(nFrames):
    meas = radon(copy.deepcopy(phantomList[iFrame])[0,:,:], iAngles) 
    if (iFrame == 0): measNoNoise = meas
    if (noise): 
        meas = sp.random.poisson(meas)
    if (iFrame == 0): measWithNoise = meas
    plt.subplot(2,nFrames/2+1,iFrame+1), plt.title('Time frame {0}'.format(iFrame)), plt.imshow(meas, interpolation=None, vmin = 0, vmax = 1000, cmap=plt.cm.Greys_r) 
    measList.append(meas) 
plt.suptitle('Measurements'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_measurements.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

plt.figure() 
plt.subplot(1,2,1), plt.title('Without noise'), plt.imshow(measNoNoise, interpolation=None, vmin = 0, vmax = noiseLevel, cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('With noise'), plt.imshow(measWithNoise, interpolation=None, vmin = 0, vmax = noiseLevel, cmap=plt.cm.Greys_r)
plt.suptitle('Time Frame 1'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_measurementsWithWithoutNoise.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

'''
reconList = []
for iFrame in range(len(measList)): 
    reconList.append(iradon(copy.deepcopy(measList[iFrame]), iAngles, filter = None)) 
    reconTMP = np.zeros((1, 460, 460))
    reconTMP[0,:,:] = reconList[iFrame]
    pyvpx.numpy2vpx(reconTMP, figSaveDir + 'reconGuess_{}.vpx'.format(iFrame)) 
guess = np.mean(reconList, axis = 0)
'''
guess = np.ones(np.shape(image2D))
plt.figure(), plt.title('Initial guess'), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(guess), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 
guessTMP = np.zeros((1,) + np.shape(image2D))
guessTMP[0,:,:] = guess
pyvpx.numpy2vpx(guessTMP, figSaveDir + 'guess.vpx') 

#_________________________NESTED EM LOOP_______________________________
offsetFoundList = []
quadErrorSumFoundList = []
quadErrorSumListList = []
guessSum = []
guessSum.append(np.sum(guess))
for iIt in range(nIt): 
    totalError = 0 
    if(iIt < 4): 
        # Normal MLEM 
        for iFrame in range(nFrames): 
            error = measList[iFrame]/radon(guess, iAngles) 
            error[np.isnan(error)] = 0
            error[np.isinf(error)] = 0
            error[error > 1E10] = 0;
            error[error < 1E-10] = 0
            errorBck = iradon(error, iAngles, filter = None) 
            totalError += errorBck
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

    if (iIt >= 4):
        # Motion model optimization
        quadErrorSumList = []   
        offsetList = range(trueOffset-5, trueOffset+6)
        for offset in offsetList: 
            quadErrorSum = 0 
            for iFrame in range(nFrames): 
                guessMoved = np.zeros(np.shape(guess))
                guessMoved = sp.ndimage.shift(copy.deepcopy(guess), (surSignal[iFrame] - offset, 0)) # Je bent als het ware de correctie op het surrogaat signaal aan het zoeken
                guessMovedProj = radon(copy.deepcopy(guessMoved), iAngles)
                quadErrorSum += np.sum(abs(guessMovedProj - measList[iFrame]))
            quadErrorSumList.append({'offset' : offset, 'quadErrorSum' : quadErrorSum})
            print 'Offset: {}'.format(offset), 'Quadratic error: {}'.format(quadErrorSum)

        quadErrorSums = [x['quadErrorSum'] for x in quadErrorSumList]
        quadErrorSumListList.append(quadErrorSums)
        index = quadErrorSums.index(np.min(quadErrorSums))
        offsetFound = quadErrorSumList[index]['offset']
        offsetFoundList.append(offsetFound)
        quadErrorSumFound = quadErrorSumList[index]['quadErrorSum']
        quadErrorSumFoundList.append(quadErrorSumFound) 

        plt.plot(offsetList, quadErrorSums, 'b-', offsetFound, quadErrorSumFound, 'ro'), plt.title('Quadratic error vs. offset TEST')
        plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError_Iteration{}.png'.format(numFigures, trueShiftAmplitude, iIt))
        numFigures += 1 
        plt.close()

        totalError = 0 
        # Normal MLEM with motion compensation 
        for iFrame in range(nFrames): 
            shiftedGuess = np.zeros(np.shape(guess))
            shiftedGuess = sp.ndimage.shift(copy.deepcopy(guess), (surSignal[iFrame] - offsetFound, 0))
            shiftedGuessSinogram = radon(shiftedGuess, iAngles) 
            error = measList[iFrame]/shiftedGuessSinogram 
            error[np.isnan(error)] = 0
            error[np.isinf(error)] = 0
            error[error > 1E10] = 0;
            error[error < 1E-10] = 0
            errorBck = iradon(error, iAngles, filter = None) 
            errorBckShifted = np.zeros(np.shape(errorBck))
            errorBckShifted = sp.ndimage.shift(errorBck, (-surSignal[iFrame] + offsetFound, 0))
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
    plt.plot(offsetFoundList, quadErrorSumFoundList, 'ro') 
    plt.plot(offsetList, quadErrorSumListList[i], label = 'Iteration {}'.format(i)), plt.title('Quadratic error vs. offset')
    plt.axvline(trueOffset, color='k', linestyle='--')
plt.legend()
plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError.png'.format(numFigures, trueShiftAmplitude))
numFigures += 1 
plt.close()