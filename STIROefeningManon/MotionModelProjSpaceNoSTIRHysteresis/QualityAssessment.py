import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import iradon, radon, rescale
import ManonsFunctions as mf 
import scipy as sp
import pyvpx
import copy
from scipy.optimize import curve_fit, minimize, brute
from scipy.signal import argrelextrema
from heapq import merge
from matplotlib.ticker import MaxNLocator
from skimage.measure import find_contours, points_in_poly

stationary = True 
#stationary = False # False is only possible for sinusoidal motion! 
modelBroken = False  
#modelBroken = False  

# Parameters that influence the figure saving directory 
phantom = 'Liver'
#phantom = 'Block'
#noise = False
noise = True
#motion = 'Step' 
motion = 'Sine'

# Create a direcotory for figure storage (just the string, make sure  the folder already exists!) 
dir = './Figures/'
figSaveDir = mf.make_figSaveDir(dir, motion, phantom, noise, stationary, modelBroken)
numFigures = 0 

#___________Quantitative analyses____________

qualityFile = open(figSaveDir + "QualityAssessment.txt", "w")

guess = pyvpx.vpx2numpy(figSaveDir + 'guess_Iteration9.vpx')
guess = guess[0,:,:]

# Target volumes 
largeTarget = guess[125:185, 100:170]
smallTarget = guess[115:175, 150:220]

plt.figure(), plt.title('Target volume (large lesion)'), plt.imshow(largeTarget, interpolation = None, vmin = 0, vmax = np.max(largeTarget), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig_largeTarget'), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(largeTarget) )
image2DTMP[0,:,:] = largeTarget
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'largeTarget.vpx') 

plt.figure(), plt.title('Target volume (snall lesion)'), plt.imshow(smallTarget, interpolation = None, vmin = 0, vmax = np.max(smallTarget), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig_smallTarget'), plt.close() 
numFigures += 1 
image2DTMP = np.zeros((1,) + np.shape(smallTarget) )
image2DTMP[0,:,:] = smallTarget
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'smallTarget.vpx') 
    
# SUV max 
largeMax = np.max(largeTarget)
smallMax = np.max(smallTarget)

# 2D Volumes 
largeThresVolume = np.zeros(np.shape(largeTarget))
for i in range(np.shape(largeTarget)[0]):
    for j in range(np.shape(largeTarget)[1]):
        if (largeTarget[i,j] > 0.5*largeMax): 
            largeThresVolume[i,j] = largeTarget[i,j]

smallThresVolume = np.zeros(np.shape(smallTarget))
for i in range(np.shape(smallTarget)[0]):
    for j in range(np.shape(smallTarget)[1]):
        if (smallTarget[i,j] > 0.5*smallMax):
            smallThresVolume[i,j] = smallTarget[i,j]

plt.figure(), plt.title('Thresholded volume (large lesion)'), plt.imshow(largeThresVolume, interpolation = None, vmin = 0, vmax = np.max(largeThresVolume), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig_largeThresholdedVolume'), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(largeThresVolume) )
image2DTMP[0,:,:] = largeThresVolume
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'largeThresholdedVolume.vpx') 

plt.figure(), plt.title('Thresholded volume (small lesion)'), plt.imshow(smallThresVolume, interpolation = None, vmin = 0, vmax = np.max(smallThresVolume), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig_smallThresholdedVolume'), plt.close() 
numFigures += 1 
image2DTMP = np.zeros((1,) + np.shape(smallThresVolume) )
image2DTMP[0,:,:] = smallThresVolume
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'smallThresholdedVolume.vpx') 

# 2D Volume contours 
largeContourReversed = find_contours(largeThresVolume, 0.001) 
maxLength = 0 
for i in range(len(largeContourReversed)):     
    if(np.shape(largeContourReversed[i]) > maxLength): 
        maxLength = np.shape(largeContourReversed[i])
        largeMaxContour = largeContourReversed[i]
largeContour = [(i[1], i[0]) for i in largeMaxContour] 
largeContourX, largeContourY = np.array(largeContour).T 

smallContourReversed = find_contours(smallThresVolume, 0.001) 
maxLength = 0 
for i in range(len(smallContourReversed)):     
    if(np.shape(smallContourReversed[i]) > maxLength): 
        maxLength = np.shape(smallContourReversed[i])
        smallMaxContour = smallContourReversed[i]
smallContour = [(i[1], i[0]) for i in smallContourReversed[3]] 
smallContourX, smallContourY = np.array(smallContour).T 

plt.figure(), plt.title('Thresholded volume with contour (large lesion)'), plt.imshow(largeThresVolume, interpolation = None, vmin = 0, vmax = np.max(largeThresVolume), cmap=plt.cm.Greys_r)
plt.scatter(largeContourX, largeContourY), plt.savefig(figSaveDir + 'QFig_largeThresholdedVolumeContour'), plt.close()
numFigures += 1  

plt.figure(), plt.title('Target volume with contour (large lesion)'), plt.imshow(largeTarget, interpolation = None, vmin = 0, vmax = np.max(largeTarget), cmap=plt.cm.Greys_r)
plt.scatter(largeContourX, largeContourY), plt.savefig(figSaveDir + 'QFig_largeTargetVolumeContour'), plt.close()
numFigures += 1  

plt.figure(), plt.title('Thresholded volume with contour (small lesion)'), plt.imshow(smallThresVolume, interpolation = None, vmin = 0, vmax = np.max(smallThresVolume), cmap=plt.cm.Greys_r)
plt.scatter(smallContourX, smallContourY), plt.savefig(figSaveDir + 'QFig_smallThresholdedVolumeContour'), plt.close()
numFigures += 1 

plt.figure(), plt.title('Target volume with contour (small lesion)'), plt.imshow(smallTarget, interpolation = None, vmin = 0, vmax = np.max(smallTarget), cmap=plt.cm.Greys_r)
plt.scatter(smallContourX, smallContourY), plt.savefig(figSaveDir + 'QFig_smallTargetVolumeContour'), plt.close()
numFigures += 1   

# Binary masks 
# Coordinates are [y,x]! 
smallNy = np.shape(smallThresVolume)[1]
smallNx = np.shape(smallThresVolume)[0]
smallTargetCoordsMatrix = [[(i,j) for i in range(smallNy)] for j in range(smallNx)] 
smallTargetCoordsVec = []
for i in range(len(smallTargetCoordsMatrix)): smallTargetCoordsVec += smallTargetCoordsMatrix[i]

smallBinMaskVec = points_in_poly(smallTargetCoordsVec, smallContour)
smallBinMaskMatrix = np.array(smallBinMaskVec).reshape(smallNx, smallNy) 
smallBinMaskMatrix = np.array(smallBinMaskMatrix).astype(int)
smallVolume = smallBinMaskMatrix*smallThresVolume 


plt.figure(), plt.title('Binary mask (large lesion)'), plt.imshow(largeBinMaskMatrix, interpolation = None, vmin = 0, vmax = np.max(largeBinMaskMatrix), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig_largeBinaryMask'), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(largeBinMaskMatrix) )
image2DTMP[0,:,:] = largeBinMaskMatrix
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'largeBinaryMask.vpx') 

largeNy = np.shape(largeThresVolume)[1]
largeNx = np.shape(largeThresVolume)[0]
largeTargetCoordsMatrix = [[(i,j) for i in range(largeNy)] for j in range(largeNx)] 
largeTargetCoordsVec = []
for i in range(len(largeTargetCoordsMatrix)): largeTargetCoordsVec += largeTargetCoordsMatrix[i]

largeBinMaskVec = points_in_poly(largeTargetCoordsVec, largeContour)
largeBinMaskMatrix = np.array(largeBinMaskVec).reshape(largeNx, largeNy) 
largeBinMaskMatrix = np.array(largeBinMaskMatrix).astype(int)
largeVolume = largeBinMaskMatrix*largeThresVolume 

plt.figure(), plt.title('Binary mask (small lesion)'), plt.imshow(smallBinMaskMatrix, interpolation = None, vmin = 0, vmax = np.max(smallBinMaskMatrix), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig_smallBinaryMask'), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(smallBinMaskMatrix) )
image2DTMP[0,:,:] = smallBinMaskMatrix
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'smallBinaryMask.vpx') 

# SUV mean 
largeMean = np.mean(largeVolume)
smallMean = np.mean(smallVolume)

# Volume sum 
largeVolumeSum = np.sum(largeBinMaskMatrix)
smallVolumeSum = np.sum(smallBinMaskMatrix)

# CNR 
largeBckVolume = guess[123:183, 125:195]*largeBinMaskMatrix 
largeCNR = abs(largeMean - np.mean(largeBckVolume))/np.std(largeBckVolume)

smallBckVolume = guess[123:183, 125:195]*smallBinMaskMatrix 
smallCNR = abs(smallMean - np.mean(smallBckVolume))/np.std(smallBckVolume)

# Write results 
qualityFile.write('SUV max L: {}\n'.format(largeMax))
qualityFile.write('SUV max S: {}\n'.format(smallMax))
qualityFile.write('SUV mean L: {}\n'.format(largeMean))
qualityFile.write('SUV mean S: {}\n'.format(smallMean))
qualityFile.write('Volume sum L: {}\n'.format(largeVolumeSum))
qualityFile.write('Volume sum S: {}\n\n'.format(smallVolumeSum))
qualityFile.write('CNR L: {}\n\n'.format(largeCNR))
qualityFile.write('CNR S: {}\n\n'.format(smallCNR))

qualityFile.close() 

'''
smallNx = np.shape(smallVolume)[1]
smallNy = np.shape(smallVolume)[0]
smallTargetCoordsMatrix = [[(i,j) for i in range(smallNx)] for j in range(smallNy)] 
smallTargetCoordsVec = []
for i in range(len(smallTargetCoordsMatrix)): smallTargetCoordsVec += smallTargetCoordsMatrix[i]

smallBinMaskVec = points_in_poly(smallTargetCoordsVec, smallContour)

smallBinMaskMatrix = np.zeros(np.shape(smallVolume))
for i in range(smallNx): 
    for j in range(smallNy): 
        if(smallBinMaskVec[i*(smallNx-1) + i + 1]): smallBinMaskMatrix[i,j] = 1 

nX = 5 
nY = 3 
testMatrix = np.zeros((nY,nX))
testCoordsMatrix = [[(i,j) for i in range(nX)] for j in range(nY)] 
testCoordsVec = [] 
for i in range(len(testCoordsMatrix)): testCoordsVec += testCoordsMatrix[i]

smallTargetCoordsVec = []
for i in range(len(smallTargetCoordsMatrix)): smallTargetCoordsVec += smallTargetCoordsMatrix[i]
binMaskVec = [False, False, True, False, False, True, False, False, True, False, False, True, False, False, True]   
binMaskMatrix = np.zeros(np.shape(smallTargetVolume))
for i in range(nX): 
    for j in range(nY): 
        if(binMaskVec[j*(nX-1)+i+1]): binMaskMatrix[i,j] = 1 
'''