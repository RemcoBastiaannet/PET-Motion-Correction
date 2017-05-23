def make_figSaveDir(dir, motion, phantom, noise, stationary):
    # Make sure all possible directories exist! 
    dir += '{}/'.format(motion)
    dir += '{}/'.format(phantom)
    dir += 'Noise_{}/'.format(noise)
    dir += 'Stationary_{}/'.format(stationary)
    return dir 

from skimage.transform import iradon, radon, rescale
from skimage import data_dir
from skimage.io import imread
import numpy as np
import math
import scipy.ndimage as spim
import copy

def make_Phantom(phantom, duration, noiseLevel): 
    if phantom == 'Block': 
        image = np.zeros((160,160))
        image[65:95, 65:95] = 1 
    elif phantom == 'Shepp-Logan': 
        imageSmall = imread(data_dir + "/phantom.png", as_grey=True)
        imageSmall = rescale(imageSmall, scale=0.2)

        tmpY = np.zeros((80, np.shape(imageSmall)[1])) 
        image = np.concatenate((tmpY, imageSmall), axis = 0)
        image = np.concatenate((image, tmpY), axis = 0)

        tmpX = np.zeros((np.shape(image)[0], 80))
        image = np.concatenate((tmpX, image), axis = 1)
        image = np.concatenate((image, tmpX), axis = 1)

    image *= noiseLevel*duration/np.sum(image) # number of total counts per second times duration divided by the sum of the image 

    return image 

def move_Phantom(motion, nFrames, trueShiftAmplitude, trueSlope, image, stationary): 
    phantomList = [] 
    shiftList = []
    Nx = np.shape(image)[1] 
    Ny = np.shape(image)[0]

    for iFrame in range(nFrames): 
        if 'Sine' in motion:
            shift = trueShiftAmplitude * math.sin(2*math.pi*iFrame/9)
            if ((not stationary) and (iFrame > nFrames/2)): 
                shift += 2*trueShiftAmplitude
        elif 'Step' in motion: 
                shift = iFrame*trueShiftAmplitude
    
        shiftList.append(shift) 

        tmp = np.zeros((1, Ny, Nx))
        tmp[0] = image      
        tmp = spim.shift(tmp, [0, shift, 0], cval = 0.0)
        tmp[tmp < 1E-10] = 0

        phantomList.append(copy.deepcopy(tmp))
    
    surSignal = [elem/trueSlope for elem in shiftList]

    return (phantomList, surSignal, shiftList) 

def write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueSlope, duration, nFrames, gating): 
    file = open(figSaveDir + "Configuratie.txt", "w")
    file.write("Phantom: {}\n".format(phantom))
    file.write("Noise: {}\n".format(noise))
    file.write("Motion: {}\n".format(motion))
    file.write("Stationary: {}\n".format(stationary)) 
    file.write("Number of iterations: {}\n".format(nIt))
    file.write("True shift amplitude: {}\n".format(trueShiftAmplitude))
    file.write("True slope (motion model): {}\n".format(trueSlope))
    file.write("Scan duration: {}\n".format(duration))
    file.write("Number of time frames: {}\n".format(nFrames))
    file.write("Gating: {}\n".format(gating))
    file.close()

def gating(nonGatedSurSignal, nonGatedPhantomList, nonGatedShiftList, gateMin, gateMax): 
    surSignal = []
    phantomList = []
    shiftList = []
    for i in range(len(nonGatedSurSignal)): 
        if ((nonGatedSurSignal[i] <= gateMax) and (nonGatedSurSignal[i] >= gateMin)): 
            surSignal.append(nonGatedSurSignal[i])
            phantomList.append(nonGatedPhantomList[i])
            shiftList.append(nonGatedShiftList[i])
    
    return (surSignal, phantomList, shiftList) 