def make_figSaveDir(motion, phantom, noise):
    # Make sure all possible directories exist! 
    dir = './Figures/'
    if (motion == 'Step'): dir += 'Step/'
    elif (motion == 'Sine'): dir += 'Sine/'
    if (phantom == 'Block'): dir += 'Block/'
    elif (phantom == 'Shepp-Logan'): dir += 'Shepp-Logan/'
    if (noise): dir += 'Noise/'
    else: dir += 'No_Noise/'
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

        tmpY = np.zeros((50, np.shape(imageSmall)[1])) 
        image = np.concatenate((tmpY, imageSmall), axis = 0)
        image = np.concatenate((image, tmpY), axis = 0)

        tmpX = np.zeros((np.shape(image)[0], 50))
        image = np.concatenate((tmpX, image), axis = 1)
        image = np.concatenate((image, tmpX), axis = 1)

    image *= 1000*duration/np.sum(image) 

    return image 

def move_Phantom(motion, nFrames, trueShiftAmplitude, trueOffset, image): 
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