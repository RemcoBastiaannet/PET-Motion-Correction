from skimage.transform import iradon, radon, rescale
from skimage import data_dir
from skimage.io import imread
import numpy as np
import math
import scipy.ndimage as spim
import copy

# Creates a string with the directory for storing images 
def make_figSaveDir(dir, motion, phantom, noise, stationary):
    # Make sure all possible directories exist! (or at least the ones that you actually intend to use) 
    dir += '{}/'.format(motion)
    dir += '{}/'.format(phantom)
    dir += 'Noise_{}/'.format(noise)
    dir += 'Stationary_{}/'.format(stationary)
    return dir 

# Creates a phantom 
def make_Phantom(phantom, noiseLevel): 
    # Block phantom 
    if phantom == 'Block': 
        image = np.zeros((160,160))
        image[65:95, 65:95] = 1 

    # Shepp-Logan phantom 
    elif phantom == 'Shepp-Logan': 
        # Read in and scale the image size 
        imageSmall = imread(data_dir + "/phantom.png", as_grey=True)
        imageSmall = rescale(imageSmall, scale=0.2)

        # Add zeros around the image to avoid problems with activity moving out of the FOV 
        tmpY = np.zeros((80, np.shape(imageSmall)[1])) 
        image = np.concatenate((tmpY, imageSmall), axis = 0)
        image = np.concatenate((image, tmpY), axis = 0)

        tmpX = np.zeros((np.shape(image)[0], 80))
        image = np.concatenate((tmpX, image), axis = 1)
        image = np.concatenate((image, tmpX), axis = 1)

    # Scale the content of the pixels of the image (the intensity/activity) 
    image *= noiseLevel/np.sum(image) 

    return image 

# Creates a surrogate signal and shifts the phantom in the x- and y-direction according to some motion model 
def move_Phantom(motion, nFrames, trueShiftAmplitude, trueSlope, trueSquareSlope, image, stationary): 
    # Lists for data storage 
    shiftList = [] # y-axis 
    shiftXList = [] # x-axis
    surSignal = [] 
    phantomList = [] 

    # Get image shapes 
    Nx = np.shape(image)[1] 
    Ny = np.shape(image)[0]

    for iFrame in range(nFrames): 
        # Sinusoidal motion 
        if 'Sine' in motion:
            # Create surrogate signal 
            phase = 2*math.pi*iFrame/19
            sur = trueShiftAmplitude * math.sin(phase) 
            
            # Create shift in the y-direction (using motion model) 
            shift = trueSlope*sur 

            # Create shift in the x-direction (using motion model), shift depends on the phase of the surrogate signal (inhale or exhale) 
            phaseMod = phase % (2*math.pi)
            if (phaseMod >= math.pi/2.0 and phaseMod < 3.0*math.pi/2.0): # Inhale 
                shiftX = trueSlope*sur + trueSquareSlope*sur**2  
            else: # Inhale 
                shiftX = trueSlope*sur - trueSquareSlope*sur**2 + 2*trueSquareSlope*trueShiftAmplitude**2   
            
            # Add non-stationarity (upwards shift) half-way through the signal 
            if ((not stationary) and (iFrame > nFrames/2)): 
                shift += 2*trueShiftAmplitude

        # Step function 
        elif 'Step' in motion: 
            sur = iFrame*trueShiftAmplitude # Surrogate signal 
            shift = trueSlope*sur # Image shift (using the motion model) 
       
        # Shift image in the y-direction
        tmp = np.zeros((1, Ny, Nx))
        tmp[0] = image      
        tmp = spim.shift(tmp, [0.0, shift, 0.0], cval = 0.0)
        tmp[tmp < 1E-10] = 0 # Because of problems with the spim.shift function that sometimes returns small negative values rather than 0, but radon can't handle negative values...
        image = tmp 

        # Shift image in the x-direction
        tmpX = np.zeros((1, Ny, Nx))
        tmpX[0] = image      
        tmpX = spim.shift(tmp, [0.0, 0.0, shiftX], cval = 0.0)
        tmpX[tmpX < 1E-10] = 0 # Because of problems with the spim.shift function that sometimes returns small negative values rather than 0, but radon can't handle negative values...
        image = tmpX 

        # Store the data in lists
        shiftList.append(shift) 
        shiftXList.append(shiftX)
        surSignal.append(sur) 
        phantomList.append(copy.deepcopy(image))

    return (phantomList, surSignal, shiftList, shiftXList) 

# Writes all parameters that can be specified for a simulation to a text file for storage 
def write_Configuration(figSaveDir, phantom, noise, motion, stationary, nIt, trueShiftAmplitude, trueSlope, trueSquareSlope, nFrames): 
    file = open(figSaveDir + "Configuratie.txt", "w")
    file.write("Phantom: {}\n".format(phantom))
    file.write("Noise: {}\n".format(noise))
    file.write("Motion: {}\n".format(motion))
    file.write("Stationary: {}\n".format(stationary)) 
    file.write("Number of iterations: {}\n".format(nIt))
    file.write("True shift amplitude: {}\n".format(trueShiftAmplitude))
    file.write("True slope (motion model): {}\n".format(trueSlope))
    file.write("True suare slope (motion model): {}\n".format(trueSquareSlope))
    file.write("Number of time frames: {}\n".format(nFrames))
    file.close()

# Takes the data, keeps only the data between gateMin and gateMax and returns the new gated data 
def gating(nonGatedSurSignal, nonGatedPhantomList, nonGatedShiftList, gateMin, gateMax): 
    # Lists for data storage 
    surSignal = []
    phantomList = []
    shiftList = []
    
    # Perform gating on the data 
    for i in range(len(nonGatedSurSignal)): 
        if ((nonGatedSurSignal[i] <= gateMax) and (nonGatedSurSignal[i] >= gateMin)): 
            surSignal.append(nonGatedSurSignal[i])
            phantomList.append(nonGatedPhantomList[i])
            shiftList.append(nonGatedShiftList[i])
    
    return (surSignal, phantomList, shiftList) 