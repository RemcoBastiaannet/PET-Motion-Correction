import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon
import ManonsFunctions as mf 
import scipy as sp
from copy import deepcopy 

#phantom = 'Block'
phantom = 'Shepp-Logan' 
noise = False
#noise = True
#motion = 'Step' 
motion = 'Sine'

nIt = 3 
trueShiftAmplitude = 30 # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd) + LET OP: kan niet groter zijn dan de lengte van het plaatje (kan de code niet aan) 
trueOffset = 20
numFigures = 0 
if (motion == 'Step'): nFrames = 2
else: nFrames = 3

figSaveDir = mf.make_figSaveDir(motion, phantom, noise)

#_________________________MAKE PHANTOM_______________________________
image2D = mf.make_Phantom(phantom)
plt.figure(), plt.title('Original image'), plt.imshow(image2D, interpolation = None, vmin = 0, vmax = 1), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1
 
#_________________________ADD MOTION_______________________________ 
phantomList, surSignal, shiftList = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueOffset, image2D)
originalImage = phantomList[0]

for i in range(nFrames):    
    plt.subplot(2,nFrames/2+1,i+1), plt.title('Time frame {0}'.format(i)), plt.imshow(phantomList[i][0,:,:], interpolation=None, vmin = 0, vmax = 1) 
plt.suptitle('Phantom'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 
 
plt.plot(range(nFrames), surSignal, label = 'Surrogate signal'), plt.title('Sinusoidal phantom shifts'), plt.xlabel('Time frame'), plt.ylabel('Shift')
plt.plot(range(nFrames), shiftList, label = 'True motion'), plt.legend(loc = 4), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_shiftList.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

'''
for i in range(nFrames):    
    plt.figure(figsize=(5.0, 5.0)), plt.title('{0}'.format(i)), plt.imshow(phantomList[i][0,:,:], interpolation=None, vmin = 0, vmax = 1), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantomFrame{}.png'.format(numFigures, trueShiftAmplitude, i)), plt.close()
    numFigures += 1   
'''

#_________________________MEASUREMENT, INITIAL GUESS, NORMALIZATION_______________________________
iAngles = np.linspace(0, 360, 120)[:-1]

measList = []
for iFrame in range(nFrames):
    meas = radon(phantomList[iFrame][0,:,:], iAngles) 
    measList.append(meas) 

reconList = []
for iFrame in range(len(measList)): 
    reconList.append(iradon(measList[iFrame], iAngles)) 
    #plt.figure(), plt.title('Recon frame {}'.format(iFrame)), plt.imshow(reconList[iFrame], interpolation = None, vmin = 0, vmax = 1), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_ReconFrame{}.png'.format(numFigures, trueShiftAmplitude, iFrame)), plt.close()
    #numFigures += 1 
guess = np.mean(reconList, axis = 0)
plt.figure(), plt.title('Initial guess'), plt.imshow(guess, interpolation = None, vmin = 0, vmax = 1), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 

normSino = np.ones(np.shape(measList[0]))
norm = iradon(normSino, iAngles, filter = None) # We willen nu geen ramp filter
plt.figure(), plt.title('MLEM normalization'), plt.imshow(norm, interpolation = None, vmin = 0, vmax = 0.03), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_norm.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1  

#_________________________NESTED EM LOOP_______________________________
offsetFoundList = []
quadErrorSumFoundList = []
quadErrorSumListList = []
for iIt in range(nIt): 
    # Normal MLEM 
    guessSinogram = radon(guess, iAngles) 
    error = measList[0]/guessSinogram # Je neemt het eerste time frame als "het" fantoom (maakt niet uit, beweging ga je hierna pas zoeken) 
    error[np.isnan(error)] = 0
    error[np.isinf(error)] = 0
    error[error > 1E10] = 0;
    error[error < 1E-10] = 0
    errorBck = iradon(error, iAngles, filter = None) 
    guess *= errorBck/norm
    countIt = iIt+1 

    # Motion model optimization
    guessMovedList = []
    guessMovedProjList = []
    quadErrorSumList = []   
    offsetList = [trueOffset-1, trueOffset, trueOffset+1] 
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

    plt.plot(offsetList, quadErrorSums, 'b-', offsetFound, quadErrorSumFound, 'ro'), plt.title('Quadratic error vs. offset TEST')
    plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_QuadraticError_Iteration{}.png'.format(numFigures, trueShiftAmplitude, iIt))
    numFigures += 1 
    plt.close()

    # Motion compensation 
    reconMovedList = []
    reconMovedCorList = [] 
    for iFrame in range(nFrames):
        reconMoved = iradon(guessMovedProjList[iFrame], iAngles) # FOUT -> het moet de ONBEWOGEN projectie van je guess zijn + je guess moet 
        reconMovedList.append(reconMoved) 
        reconMovedCorList.append(np.zeros(np.shape(reconMoved))) 
        sp.ndimage.shift(reconMoved, (-surSignal[iFrame] + offsetFound, 0), reconMovedCorList[iFrame])
    
    guess = np.mean(reconMovedCorList, axis = 0)

plt.figure(), plt.title('Guess after {0} iteration(s)'.format(iIt+1)), plt.imshow(guess, interpolation = None, vmin = 0, vmax = 1), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_finalImage.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1  

plt.figure(), plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(originalImage[0,:,:], interpolation=None, vmin = 0, vmax = 1)
plt.subplot(1,2,2), plt.title('Reconstructed Image'), plt.imshow(guess, interpolation=None, vmin = 0, vmax = 1), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_originalAndRecon.png'.format(numFigures, trueShiftAmplitude)), plt.close() 
numFigures += 1 