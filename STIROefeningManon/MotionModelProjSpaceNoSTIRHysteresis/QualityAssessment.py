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

### _______ LET OP !! _______
# Pas ook het bestandspad aan waar guess vandaan wordt gehaald 

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
figSaveDir += 'Kwantitatief/'
numFigures = 0 

#___________Quantitative analyses____________

qualityFile = open(figSaveDir + 'QualityAssessment.txt', "w")

guess = pyvpx.vpx2numpy('E:/Manon/Resultaten_Simulaties/1_Geen_beweging_(referentie)/guess_Iteration9.vpx') 
guess = guess[0,:,:]

# Target volumes 
largeTarget = guess[125:185, 100:170]
smallTarget = guess[115:175, 150:220]

plt.figure(), plt.title('Target volume (large lesion)'), plt.imshow(largeTarget, interpolation = None, vmin = 0, vmax = np.max(largeTarget), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig{}_largeTarget'.format(numFigures)), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(largeTarget) )
image2DTMP[0,:,:] = largeTarget
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'largeTarget.vpx') 

plt.figure(), plt.title('Target volume (snall lesion)'), plt.imshow(smallTarget, interpolation = None, vmin = 0, vmax = np.max(smallTarget), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig{}_smallTarget'.format(numFigures)), plt.close() 
numFigures += 1 
image2DTMP = np.zeros((1,) + np.shape(smallTarget) )
image2DTMP[0,:,:] = smallTarget
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'smallTarget.vpx') 
    
# SUV max 
largeMax = np.max(largeTarget)
smallMax = np.max(smallTarget)

# 2D Thresholded volumes 
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

plt.figure(), plt.title('Thresholded volume (large lesion)'), plt.imshow(largeThresVolume, interpolation = None, vmin = 0, vmax = np.max(largeThresVolume), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig{}_largeThresholdedVolume'.format(numFigures)), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(largeThresVolume) )
image2DTMP[0,:,:] = largeThresVolume
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'largeThresholdedVolume.vpx') 

plt.figure(), plt.title('Thresholded volume (small lesion)'), plt.imshow(smallThresVolume, interpolation = None, vmin = 0, vmax = np.max(smallThresVolume), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig{}_smallThresholdedVolume'.format(numFigures)), plt.close() 
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
smallContour = [(i[1], i[0]) for i in smallMaxContour] 
smallContourX, smallContourY = np.array(smallContour).T 

plt.figure(), plt.title('Thresholded volume with contour (large lesion)'), plt.imshow(largeThresVolume, interpolation = None, vmin = 0, vmax = np.max(largeThresVolume), cmap=plt.cm.Greys_r)
plt.scatter(largeContourX, largeContourY), plt.savefig(figSaveDir + 'QFig{}_largeThresholdedVolumeContour'.format(numFigures)), plt.close()
numFigures += 1  

plt.figure(), plt.title('Target volume with contour (large lesion)'), plt.imshow(largeTarget, interpolation = None, vmin = 0, vmax = np.max(largeTarget), cmap=plt.cm.Greys_r)
plt.scatter(largeContourX, largeContourY), plt.savefig(figSaveDir + 'QFig{}_largeTargetVolumeContour'.format(numFigures)), plt.close()
numFigures += 1  

plt.figure(), plt.title('Thresholded volume with contour (small lesion)'), plt.imshow(smallThresVolume, interpolation = None, vmin = 0, vmax = np.max(smallThresVolume), cmap=plt.cm.Greys_r)
plt.scatter(smallContourX, smallContourY), plt.savefig(figSaveDir + 'QFig{}_smallThresholdedVolumeContour'.format(numFigures)), plt.close()
numFigures += 1 

plt.figure(), plt.title('Target volume with contour (small lesion)'), plt.imshow(smallTarget, interpolation = None, vmin = 0, vmax = np.max(smallTarget), cmap=plt.cm.Greys_r)
plt.scatter(smallContourX, smallContourY), plt.savefig(figSaveDir + 'QFig{}_smallTargetVolumeContour'.format(numFigures)), plt.close()
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
smallVolume = smallBinMaskMatrix*smallTarget

largeNy = np.shape(largeThresVolume)[1]
largeNx = np.shape(largeThresVolume)[0]
largeTargetCoordsMatrix = [[(i,j) for i in range(largeNy)] for j in range(largeNx)] 
largeTargetCoordsVec = []
for i in range(len(largeTargetCoordsMatrix)): largeTargetCoordsVec += largeTargetCoordsMatrix[i]

largeBinMaskVec = points_in_poly(largeTargetCoordsVec, largeContour)
largeBinMaskMatrix = np.array(largeBinMaskVec).reshape(largeNx, largeNy) 
largeBinMaskMatrix = np.array(largeBinMaskMatrix).astype(int)
largeVolume = largeBinMaskMatrix*largeTarget

plt.figure(), plt.title('Binary mask (small lesion)'), plt.imshow(smallBinMaskMatrix, interpolation = None, vmin = 0, vmax = np.max(smallBinMaskMatrix), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig{}_smallBinaryMask'.format(numFigures)), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(smallBinMaskMatrix) )
image2DTMP[0,:,:] = smallBinMaskMatrix
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'smallBinaryMask.vpx') 

plt.figure(), plt.title('Binary mask (large lesion)'), plt.imshow(largeBinMaskMatrix, interpolation = None, vmin = 0, vmax = np.max(largeBinMaskMatrix), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig{}_largeBinaryMask'.format(numFigures)), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(largeBinMaskMatrix) )
image2DTMP[0,:,:] = largeBinMaskMatrix
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'largeBinaryMask.vpx') 

# SUV mean 
largeMean = np.mean(largeVolume[largeVolume != 0])
smallMean = np.mean(smallVolume[smallVolume != 0])

# Volume sum 
largeVolumeSum = np.sum(largeBinMaskMatrix)
smallVolumeSum = np.sum(smallBinMaskMatrix)

# Background 
extendedLargeBinMaskMatrix = np.zeros(np.shape(guess))
extendedLargeBinMaskMatrix[125:185, 100:170] = largeBinMaskMatrix

extendedSmallBinMaskMatrix = np.zeros(np.shape(guess))
extendedSmallBinMaskMatrix[115:175, 150:220] = smallBinMaskMatrix

totalBinMaskMatrix = extendedLargeBinMaskMatrix + extendedSmallBinMaskMatrix

plt.figure(), plt.title('Binary mask both lesions'), plt.imshow(totalBinMaskMatrix, interpolation = None, vmin = 0, vmax = np.max(totalBinMaskMatrix), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig{}_totalBinMask'.format(numFigures)), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(totalBinMaskMatrix) )
image2DTMP[0,:,:] = totalBinMaskMatrix
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'totalBinaryMask.vpx') 

bckTarget = guess*(1-totalBinMaskMatrix) 

plt.figure(), plt.title('Target volume (background)'), plt.imshow(bckTarget, interpolation = None, vmin = 0, vmax = np.max(bckTarget), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig{}_backgroundTarget'.format(numFigures)), plt.close() 
numFigures += 1 
image2DTMP = np.zeros((1,) + np.shape(bckTarget) )
image2DTMP[0,:,:] = bckTarget
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'backgroundTarget.vpx') 

bckMax = np.max(bckTarget)

bckThresVolume = np.zeros(np.shape(bckTarget))
for i in range(np.shape(bckTarget)[0]):
    for j in range(np.shape(bckTarget)[1]):
        if (bckTarget[i,j] > 0.1*bckMax): 
            bckThresVolume[i,j] = bckTarget[i,j]

plt.figure(), plt.title('Thresholded volume (background)'), plt.imshow(bckThresVolume, interpolation = None, vmin = 0, vmax = np.max(bckThresVolume), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig{}_bckThresholdedVolume'.format(numFigures)), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(bckThresVolume) )
image2DTMP[0,:,:] = bckThresVolume
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'bckThresholdedVolume.vpx') 

bckContourReversed = find_contours(bckThresVolume, 0.001)
maxLength = 0 
for i in range(len(bckContourReversed)):     
    if(np.shape(bckContourReversed[i]) > maxLength): 
        maxLength = np.shape(bckContourReversed[i])
        bckMaxContour = bckContourReversed[i]
bckContour = [(i[1], i[0]) for i in bckMaxContour] 
bckContourX, bckContourY = np.array(bckContour).T 

plt.figure(), plt.title('Thresholded volume with contour (background)'), plt.imshow(bckThresVolume, interpolation = None, vmin = 0, vmax = np.max(bckThresVolume), cmap=plt.cm.Greys_r)
plt.scatter(bckContourX, bckContourY), plt.savefig(figSaveDir + 'QFig{}_bckThresholdedVolumeContour'.format(numFigures)), plt.close()
numFigures += 1  

plt.figure(), plt.title('Target volume with contour (background)'), plt.imshow(bckTarget, interpolation = None, vmin = 0, vmax = np.max(bckTarget), cmap=plt.cm.Greys_r)
plt.scatter(bckContourX, bckContourY), plt.savefig(figSaveDir + 'QFig{}_bckTargetVolumeContour'.format(numFigures)), plt.close()
numFigures += 1 

bckNy = np.shape(bckThresVolume)[1]
bckNx = np.shape(bckThresVolume)[0]
bckTargetCoordsMatrix = [[(i,j) for i in range(bckNy)] for j in range(bckNx)] 
bckTargetCoordsVec = []
for i in range(len(bckTargetCoordsMatrix)): bckTargetCoordsVec += bckTargetCoordsMatrix[i]

bckBinMaskVec = points_in_poly(bckTargetCoordsVec, bckContour)
bckBinMaskMatrix = np.array(bckBinMaskVec).reshape(bckNx, bckNy) 
bckBinMaskMatrix = np.array(bckBinMaskMatrix).astype(int)
bckVolume = bckBinMaskMatrix*bckTarget 

plt.figure(), plt.title('Binary mask (background)'), plt.imshow(bckBinMaskMatrix, interpolation = None, vmin = 0, vmax = np.max(bckBinMaskMatrix), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig{}_bckBinaryMask'.format(numFigures)), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(bckBinMaskMatrix) )
image2DTMP[0,:,:] = bckBinMaskMatrix
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'bckBinaryMask.vpx') 

plt.figure(), plt.title('Background'), plt.imshow(bckVolume, interpolation = None, vmin = 0, vmax = np.max(bckVolume), cmap=plt.cm.Greys_r), plt.savefig(figSaveDir + 'QFig{}_backgroundVolume'.format(numFigures)), plt.close()
numFigures += 1  
image2DTMP = np.zeros((1,) + np.shape(bckVolume) )
image2DTMP[0,:,:] = bckVolume
pyvpx.numpy2vpx(image2DTMP, figSaveDir + 'backgroundVolume.vpx') 

bckMean = np.mean(bckVolume[bckVolume != 0]) 
bckVolumeSum = np.sum(bckBinMaskMatrix)
bckStd = np.std(bckVolume[bckVolume != 0])

# CNR 
largeCNR = abs(largeMean - bckMean)/bckStd

#smallBckVolume = guess[123:183, 125:195]*smallBinMaskMatrix 
smallBckVolume = guess[150:210, 100:170]*smallBinMaskMatrix 
smallCNR = abs(smallMean - bckMean)/bckStd

# Write results 
qualityFile.write('SUV max L: {}\n'.format(largeMax))
qualityFile.write('SUV mean L: {}\n'.format(largeMean))
qualityFile.write('Volume sum L: {}\n'.format(largeVolumeSum))
qualityFile.write('CNR L: {}\n\n'.format(largeCNR))

qualityFile.write('SUV max S: {}\n'.format(smallMax))
qualityFile.write('SUV mean S: {}\n'.format(smallMean))
qualityFile.write('Volume sum S: {}\n'.format(smallVolumeSum))
qualityFile.write('CNR S: {}\n\n'.format(smallCNR))

qualityFile.write('SUV max B: {}\n'.format(bckMax))
qualityFile.write('SUV mean B: {}\n'.format(bckMean))
qualityFile.write('Volume sum B: {}\n\n'.format(bckVolumeSum))

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