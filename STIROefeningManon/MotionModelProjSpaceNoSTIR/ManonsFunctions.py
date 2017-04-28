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
def make_Phantom(phantom, duration): 
    if phantom == 'Block': 
        image = np.zeros((160,160))
        image[65:95, 65:95] = 1 
    elif phantom == 'Shepp-Logan': 
        imageSmall = imread(data_dir + "/phantom.png", as_grey=True)
        imageSmall = rescale(imageSmall, scale=0.4)

        tmpY = np.zeros((150, np.shape(imageSmall)[1])) 
        image = np.concatenate((tmpY, imageSmall), axis = 0)
        image = np.concatenate((image, tmpY), axis = 0)

        tmpX = np.zeros((np.shape(image)[0], 150))
        image = np.concatenate((tmpX, image), axis = 1)
        image = np.concatenate((image, tmpX), axis = 1)

    image *= 1000*duration/np.sum(image) 

    return image 

def move_Phantom(motion, nFrames, trueShiftAmplitude, trueOffset, image, stationary): 
    phantomList = [] 
    Nx = np.shape(image)[1] 
    Ny = np.shape(image)[0]

    if (motion == 'Step'): 
        shiftList = []
        for iFrame in range(nFrames): 
            shift = iFrame*trueShiftAmplitude + trueOffset 
            shiftList.append(shift) 
            tmp = np.zeros((1, Ny, Nx))
            tmp[0] = image  

            if shift > 0: 
                tmp[0, shift:Ny, :] = tmp[0, 0:(Ny-shift), :]
                tmp[0, 0:shift, :] = 0
       
            if shift < 0: 
                tmp[0, 0:(Ny+shift), :] = tmp[0, (-shift):Ny, :] 
                tmp[0, (Ny+shift):Ny, :] = 0

            surSignal = [shiftList[i] + trueOffset for i in range(len(shiftList))]
            phantomList.append(tmp) 

    if (motion == 'Sine'):
        shiftList = [] 
        for iFrame in range(nFrames): 
            shift = int(trueShiftAmplitude * math.sin(2*math.pi*iFrame/9))
            if ((not stationary) and (iFrame > nFrames/2)): 
                shift += int(0.5*trueShiftAmplitude) 
            shiftList.append(shift) 
            tmp = np.zeros((1, Ny, Nx))
            tmp[0] = image  
    
            if shift > 0: 
                tmp[0, shift:Ny, :] = tmp[0, 0:(Ny-shift), :]
                tmp[0, 0:shift, :] = 0
       
            if shift < 0: 
                tmp[0, 0:(Ny+shift), :] = tmp[0, (-shift):Ny, :]
                tmp[0, (Ny+shift):Ny, :] = 0

            phantomList.append(tmp) 
            surSignal = [shiftList[i] + trueOffset for i in range(len(shiftList))]

    return (phantomList, surSignal, shiftList) 

def write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueOffset, duration, nFrames, gating): 
    file = open(figSaveDir + "Configuratie.txt", "w")
    file.write("Phantom: {}\n".format(phantom))
    file.write("Noise: {}\n".format(noise))
    file.write("Motion: {}\n".format(motion))
    file.write("Stationary: {}\n".format(stationary)) 
    file.write("Number of iterations: {}\n".format(nIt))
    file.write("True shift amplitude: {}\n".format(trueShiftAmplitude))
    file.write("True offset (motion model): {}\n".format(trueOffset))
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