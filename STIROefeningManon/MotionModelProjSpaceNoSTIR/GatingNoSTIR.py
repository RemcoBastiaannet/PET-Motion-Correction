import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon
import ManonsFunctions as mf 
import scipy as sp


#_________________________CONFIGURATION_______________________________
phantom = 'Shepp-Logan' 
noise = False
#noise = True
#motion = 'Step' 
motion = 'Sine'
stationary = True 
#stationary = False # Only possible for sinusoidal motion 

nIt = 10
trueShiftAmplitude = 15 # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd) + LET OP: kan niet groter zijn dan de lengte van het plaatje (kan de code niet aan) 
trueOffset = 5
numFigures = 0 
duration = 60 # in seconds
if (motion == 'Step'): nFrames = 2
else: nFrames = 10

dir = './Figures/'
figSaveDir = mf.make_figSaveDir(dir, motion, phantom, noise, stationary)
figSaveDir += 'Gating/'

mf.write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueOffset, duration, nFrames)


#_________________________MAKE PHANTOM_______________________________
image2D = mf.make_Phantom(phantom, duration)
plt.figure(), plt.title('Original image'), plt.imshow(image2D, cmap=plt.cm.Greys_r, interpolation = None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_phantom.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1


#_________________________ADD MOTION_______________________________ 
phantomList, surSignal, shiftList = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueOffset, image2D, stationary,)

# Visualization of gating 
x = np.arange(0, len(phantomList), 0.1)
plt.plot(range(len(surSignal)), surSignal, 'bo', label = 'Surrogate signal', markersize = 3), plt.title('Sinusoidal phantom shifts'), plt.xlabel('Time frame'), plt.ylabel('Shift')
plt.plot(range(len(shiftList)), shiftList, 'ro', label = 'True motion', markersize = 3) 
plt.axhline(y = gateMin, color = 'grey', label = 'Respiratory gating')
plt.axhline(y = gateMax, color = 'grey')
plt.axis([0, len(phantomList), -trueShiftAmplitude - 5, trueShiftAmplitude + trueOffset])
plt.fill_between(x, gateMin, gateMax, color='grey', alpha='0.5')
plt.legend(loc = 0), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_shiftList.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 


#_________________________MEASUREMENT_______________________________
iAngles = np.linspace(0, 360, 120)[:-1]

measList = []
for iFrame in range(len(phantomList)):
    meas = radon(phantomList[iFrame][0,:,:], iAngles) 
    if (iFrame == 0): measNoNoise = meas
    if (noise): 
        meas = sp.random.poisson(meas)
    if (iFrame == 0): measWithNoise = meas
    measList.append(meas) 

plt.subplot(1,2,1), plt.title('Without noise'), plt.imshow(measNoNoise, cmap=plt.cm.Greys_r, interpolation=None, vmin = 0, vmax = 1000)
plt.subplot(1,2,2), plt.title('With noise'), plt.imshow(measWithNoise, cmap=plt.cm.Greys_r, interpolation=None, vmin = 0, vmax = 1000)
plt.suptitle('Time Frame 1'), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_measurementsWithWithoutNoise.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 


#_________________________INITIAL GUESS_______________________________
reconList = []
for iFrame in range(len(measList)): 
    reconList.append(iradon(measList[iFrame], iAngles)) 
guess = np.mean(reconList, axis = 0)
plt.figure(), plt.title('Initial guess'), plt.imshow(guess, cmap=plt.cm.Greys_r, interpolation = None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude)), plt.close() 
numFigures += 1


#_________________________NORMALIZATION_______________________________
normSino = np.ones(np.shape(measList[0]))
norm = iradon(normSino, iAngles, filter = None) # We willen nu geen ramp filter


#_________________________NESTED EM LOOP_______________________________
guessSum = []
guessSum.append(np.sum(guess))
for iIt in range(nIt): 
    for iFrame in range(len(surSignal)): 
        shiftedGuess = np.zeros(np.shape(guess))
        shiftedGuess = guess 
        shiftedGuessSinogram = radon(shiftedGuess, iAngles) 
        error = measList[iFrame]/shiftedGuessSinogram 
        error[np.isnan(error)] = 0
        error[np.isinf(error)] = 0
        error[error > 1E10] = 0;
        error[error < 1E-10] = 0
        errorBck = iradon(error, iAngles, filter = None) 
        errorBckShifted = errorBck
        guess *= errorBckShifted
    guess /= norm 
    guessSum.append(np.sum(guess))
    countIt = iIt+1 

    plt.figure(), plt.title('Guess after {0} iteration(s)'.format(iIt+1)), plt.imshow(guess, cmap=plt.cm.Greys_r, interpolation = None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_finalImage.png'.format(numFigures, trueShiftAmplitude)), plt.close()
    numFigures += 1  

plt.figure(), plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(image2D, cmap=plt.cm.Greys_r, interpolation=None, vmin = 0, vmax = np.max(image2D))
plt.subplot(1,2,2), plt.title('Reconstructed Image'), plt.imshow(guess, cmap=plt.cm.Greys_r, interpolation=None, vmin = 0, vmax = np.max(image2D)), plt.savefig(figSaveDir + 'Fig{}_TrueShift{}_originalAndRecon.png'.format(numFigures, trueShiftAmplitude)), plt.close() 
numFigures += 1 