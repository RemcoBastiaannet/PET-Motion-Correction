def make_figSaveDir(motion, phantom, noise, stationary, gating):
    # Make sure all possible directories exist! 
    dir = './Figures/'
    if (motion == 'Step'): dir += 'Step/'
    elif (motion == 'Sine'): dir += 'Sine/'
    if (phantom == 'Block'): dir += 'Block/'
    elif (phantom == 'Shepp-Logan'): dir += 'Shepp-Logan/'
    if (noise): dir += 'Noise/'
    else: dir += 'No_Noise/'
    if (stationary): dir += 'Stationary/'
    else: dir += 'Non_Stationary/'
    if (gating): dir += 'Gating/'
    else: dir += 'No_Gating/'
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

def move_Phantom(motion, nFrames, trueShiftAmplitude, trueOffset, image, stationary, gating): 
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
        surSignal = []
        drift = -int(0.5*trueShiftAmplitude)
        for iFrame in range(nFrames): 
            shift = int(trueShiftAmplitude * math.sin(2*math.pi*iFrame/19))
            if ((not stationary) and (iFrame > nFrames/2)): 
                shift += drift 

            tmp = np.zeros((1, Ny, Nx))
            tmp[0] = image  
    
            if shift > 0: 
                tmp[0, shift:Ny, :] = tmp[0, 0:(Ny-shift), :]
                tmp[0, 0:shift, :] = 0
       
            if shift < 0: 
                tmp[0, 0:(Ny+shift), :] = tmp[0, (-shift):Ny, :]
                tmp[0, (Ny+shift):Ny, :] = 0

            phase = shift + trueOffset
            minSurSignal = -trueShiftAmplitude + trueOffset
            maxSurSignal = trueShiftAmplitude + trueOffset
            if (drift >= 0): maxSurSignal += drift      
            else: minSurSignal += drift 
            gateMin = minSurSignal
            gateMax = minSurSignal + 0.35*(maxSurSignal - minSurSignal)
            if (gating): 
                if ((phase <= gateMax) and (phase >= gateMin)): 
                    phantomList.append(tmp) 
                    surSignal.append(phase) 
                    shiftList.append(shift) 
            else: 
                phantomList.append(tmp) 
                surSignal.append(phase)
                shiftList.append(shift) 

    return (phantomList, surSignal, shiftList, gateMin, gateMax) 

def write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueOffset, duration, nFrames, gating): 
    file = open(figSaveDir + "Configuration.txt", "w")
    file.write("Phantom: {}\n".format(phantom))
    file.write("Noise: {}\n".format(noise))
    file.write("Motion: {}\n".format(motion))
    file.write("Stationary: {}\n".format(stationary)) 
    file.write("Respiratory gating: {}\n".format(gating)) 
    file.write("Number of iterations: {}\n".format(nIt))
    file.write("True shift amplitude: {}\n".format(trueShiftAmplitude))
    file.write("True offset (motion model): {}\n".format(trueOffset))
    file.write("Scan duration: {}\n".format(duration))
    file.write("Number of time frames: {}\n".format(nFrames))
    file.close()