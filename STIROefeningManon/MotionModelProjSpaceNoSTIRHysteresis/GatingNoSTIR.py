import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon
import ManonsFunctions as mf 
import scipy as sp
import copy 
import pyvpx 


#_________________________CONFIGURATION_______________________________
phantom = 'Liver' 
noise = True
motion = 'Sine'
stationary = False 
modelBroken = False 

nIt = 10
trueShiftAmplitude = 5 # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd) + LET OP: kan niet groter zijn dan de lengte van het plaatje (kan de code niet aan) 
trueSlope = 1.4 # y-axis trueSlope = 0.5 # y-axis 
trueSlopeX = 0.2 # x-axis 
numFigures = 0 
if (motion == 'Step'): nFrames = 2
else: nFrames = 18
noiseLevel = 200
x0 = np.array([1.0,1.0])
mu = 9.687E-2 # water, cm^-1
mu /= 5 # number of pixels in 1 cm if one pixel is 2 mm 

iAngles = np.linspace(0, 360, 120)[:-1]

gateNumber = 5 # possible values: 1-5

dir = './Figures/'
figSaveDir = mf.make_figSaveDir(dir, motion, phantom, noise, stationary, modelBroken)
figSaveDir += 'Gating/'

mf.write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueSlope, nFrames, x0, modelBroken)


#_________________________MAKE PHANTOM_______________________________
image2D = mf.make_Phantom(phantom, noiseLevel)
plt.figure(), plt.axis('off'), plt.title('Original image'), plt.imshow(image2D, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_phantom.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1

nX = int(np.shape(image2D)[0]) # Not sure if x and y are actually x and y 
nY = int(np.shape(image2D)[1])
imageMuBinary = copy.deepcopy(image2D)
for iX in range(nX): 
    for iY in range(nY): 
        if imageMuBinary[iX, iY] == 0: continue 
        else: imageMuBinary[iX, iY] /= image2D[iX, iY] 
imageMu = imageMuBinary*mu
sinoMu = radon(copy.deepcopy(imageMu), iAngles) 
expSinoMu = np.zeros(np.shape(sinoMu))
for i in range(np.shape(sinoMu)[0]):
    for j in range(np.shape(sinoMu)[1]):
        expSinoMu[i,j] = math.exp(-sinoMu[i,j])


#_________________________ADD MOTION_______________________________ 
nonGatedPhantomList, nonGatedSurSignal, nonGatedShiftList, nonGatedShiftListX = mf.move_Phantom(motion, nFrames, trueShiftAmplitude, trueSlope, trueSlopeX, image2D, stationary, modelBroken)


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
measList = []
for iFrame in range(len(phantomList)):
    meas = radon(phantomList[iFrame][0,:,:], iAngles) 
    if (iFrame == 0): measNoNoise = meas
    if (noise): 
        meas = sp.random.poisson(meas)
    if (iFrame == 0): measWithNoise = meas
    meas = meas.astype(np.float) * expSinoMu 
    measList.append(meas) 

plt.figure(), plt.axis('off')
plt.subplot(1,2,1), plt.title('Without noise'), plt.imshow(measNoNoise, interpolation=None, vmin = 0, vmax = np.max(measWithNoise), cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('With noise'), plt.imshow(measWithNoise, interpolation=None, vmin = 0, vmax =  np.max(measWithNoise), cmap=plt.cm.Greys_r)
plt.suptitle('Time Frame 1'), plt.savefig(figSaveDir + 'Fig{}_measurementsWithWithoutNoise.png'.format(numFigures, trueShiftAmplitude)), plt.close()
numFigures += 1 


#_________________________INITIAL GUESS_______________________________
guess = np.ones(np.shape(phantomList[0]))[0,:,:]
plt.figure(), plt.axis('off'), plt.title('Initial guess'), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_InitialGuess.png'.format(numFigures, trueShiftAmplitude)), plt.close() 
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

    plt.figure(), plt.axis('off'), plt.title('Guess after {0} iteration(s)'.format(iIt+1)), plt.imshow(guess, interpolation = None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_finalImage.png'.format(numFigures, trueShiftAmplitude)), plt.close()
    numFigures += 1  
    image2DTMP = np.zeros((1,) + np.shape(guess) )
    image2DTMP[0,:,:] = guess
    pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'guess_Iteration{}.vpx'.format(iIt)) 

plt.figure(), plt.axis('off'), plt.subplot(1,2,1), plt.title('Original Image'), plt.imshow(image2D, interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r)
plt.subplot(1,2,2), plt.title('Reconstructed Image'), plt.imshow(guess, interpolation=None, vmin = 0, vmax = np.max(image2D), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'Fig{}_originalAndRecon.png'.format(numFigures, trueShiftAmplitude)), plt.close() 
numFigures += 1 

mf.writeMhdFile(guess, figSaveDir + 'Gate{}.mhd'.format(gateNumber))