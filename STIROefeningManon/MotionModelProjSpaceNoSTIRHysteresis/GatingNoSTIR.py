import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon
import ManonsFunctionsHysteresis as mf 
import scipy as sp


#_________________________CONFIGURATION_______________________________
#phantom = 'Block'
phantom = 'Shepp-Logan' 
#noise = False
noise = True
#motion = 'Step' 
motion = 'Sine'
#stationary = True 
stationary = False # Only possible for sinusoidal motion 
hysteresis = False

nIt = 25
trueShiftAmplitude = 10 # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd) + LET OP: kan niet groter zijn dan de lengte van het plaatje (kan de code niet aan) 
trueSlope = 2.0 # y-axis trueSlope = 0.5 # y-axis 
trueSlopeX = 0.2 # x-axis 
trueSlopeInhale = 1.0 # hysteresis, x-axis
trueSlopeExhale = trueSlopeInhale # hysteresis, x-axis, must be the same as trueSlopeInhale, otherwise the two functions do are not equal at the endpoints
trueSquareSlopeInhale = +0.1 # hysteresis, x-axis
trueSquareSlopeExhale = -0.06 # hysteresis, x-axis
numFigures = 0 
if (motion == 'Step'): nFrames = 2
else: nFrames = 18
noiseLevel = 600
x0 = np.array([1.0,1.0])

gateNumber = 4 # possible values: 1-5

dir = './Figures/'
figSaveDir = mf.make_figSaveDir(dir, motion, phantom, noise, stationary)
figSaveDir += 'Gating/'

mf.write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueSlope, trueSlopeInhale, trueSlopeExhale, trueSquareSlopeInhale, trueSquareSlopeExhale, nFrames, hysteresis, x0)


#_________________________MAKE PHANTOM_______________________________
image2D = mf.make_Phantom(phantom, noiseLevel)
plt.figure(), plt.title('Original image'), plt.imshow(image2D, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_phantom.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1


#_________________________ADD MOTION_______________________________ 
nonGatedPhantomList, nonGatedSurSignal, nonGatedShiftList, nonGatedShiftListX = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueSlope, trueSlopeX, trueSlopeInhale, trueSlopeExhale, trueSquareSlopeInhale, trueSquareSlopeExhale, image2D, stationary, hysteresis)


#_________________________GATING_______________________________ 
maxSurSignal = np.max(nonGatedSurSignal) 
minSurSignal = np.min(nonGatedSurSignal)
gateMin = minSurSignal + 0.2*(gateNumber-1)*(maxSurSignal - minSurSignal)
gateMax = minSurSignal + 0.2*(gateNumber)*(maxSurSignal - minSurSignal)
surSignal, phantomList, shiftList = mf.gating(nonGatedSurSignal, nonGatedPhantomList, nonGatedShiftList, gateMin, gateMax)

# Visualization of gating 
x = np.arange(0, len(nonGatedPhantomList), 0.1)
plt.plot(range(len(nonGatedShiftList)), nonGatedShiftList, 'ro', label = 'True motion', markersize = 3), plt.title('Sinusoidal phantom shifts'), plt.xlabel('Time frame'), plt.ylabel('Shift')
plt.plot(range(len(nonGatedSurSignal)), nonGatedSurSignal, 'bo', label = 'Surrogate signal', markersize = 3)
plt.axhline(y = gateMin, color = 'grey', label = 'Respiratory gating')
plt.axhline(y = gateMax, color = 'grey')
plt.axis([0, len(nonGatedPhantomList), np.min((minSurSignal, minSurSignal*trueSlope))-2, np.max((maxSurSignal, maxSurSignal*trueSlope))+2])
plt.fill_between(x, gateMin, gateMax, color='grey', alpha='0.5')
plt.legend(loc = 0), plt.savefig(figSaveDir + 'Fig{}_shiftList.png'.format(numFigures, trueShiftAmplitude)), plt.close()
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

plt.figure() 
plt.subplot(1,2,1), plt.title('Without noise'), plt.imshow(measNoNoise, interpolation=None, vmin = 0, vmax = np.max(measWithNoise), cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('With noise'), plt.imshow(measWithNoise, interpolation=None, vmin = 0, vmax =  np.max(measWithNoise), cmap=plt.cm.Greys_r)
plt.suptitle('Time Frame 1'), plt.savefig(figSaveDir + 'Fig{}_measurementsWithWithoutNoise.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 


#_________________________INITIAL GUESS_______________________________
guess = np.ones(np.shape(phantomList[0]))[0,:,:]
plt.figure(), plt.title('Initial guess'), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude)), plt.close() 
numFigures += 1


#_________________________NESTED EM LOOP_______________________________
for iIt in range(nIt): 
    totalError = 0.0
    for iFrame in range(len(surSignal)): 
        guessSinogram = radon(guess, iAngles)
        error = measList[iFrame]/guessSinogram 
        error[np.isnan(error)] = 0
        error[np.isinf(error)] = 0
        error[error > 1E10] = 0;
        error[error < 1E-10] = 0
        errorBck = iradon(error, iAngles, filter = None) 
        totalError += errorBck 
    guess *= totalError/nFrames
    guess /= np.sum(guess) 
    guess *= np.sum(measList[-1])/np.shape(measList[-1])[1]  
    countIt = iIt+1 

    plt.figure(), plt.title('Guess after {0} iteration(s)'.format(iIt+1)), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_finalImage.png'.format(numFigures, trueShiftAmplitude)), plt.close()
    numFigures += 1  

plt.figure(), plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(image2D, interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('Reconstructed Image'), plt.imshow(guess, interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_originalAndRecon.png'.format(numFigures, trueShiftAmplitude)), plt.close() 
numFigures += 1 

mf.writeMhdFile(guess, figSaveDir + 'Gate{}.mhd'.format(gateNumber))

qualityFile = open(figSaveDir + "Quality.txt", "w")
qualityFile.write('Phantom:')
qualityFile.write('Maximum value: {}\n\n'.format(np.max(image2D))) 
qualityFile.write('Simulations:\n')
qualityFile.write('Maximum value: {}\n\n'.format(np.max(guess))) 
qualityFile.write('Quadratic difference of simulation and phantom: {}\n'.format(np.sum( (guess - image2D)**2 )))
qualityFile.close() 