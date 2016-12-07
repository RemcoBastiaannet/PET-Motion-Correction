import sys
import stir
import stirextra
import pylab
import numpy as np
import os
import time
import scipy as sp
import scipy.ndimage as ndimage

import pyvpx

def fillStirSpace(stirvol, pyvol):
    try:
        minTang = stirvol.get_min_tangential_pos_num()
        minAx = stirvol.get_max_axial_pos_num()
        minView = stirvol.get_min_view_num()

        for (iAx, iView, iTang), val in np.ndenumerate(pyvol):
            ixAx = minAx + iAx
            ixView = minView + iView
            ixTang = minTang + iTang
            stirvol[ixAx, ixView, ixTang] = val

    except(AttributeError):
        minX = stirvol.get_min_x()
        minY = stirvol.get_min_y()
        minZ = stirvol.get_min_z()
    
        for (iZ, iY, iX), val in np.ndenumerate(pyvol):
            ixX = minX + iX
            ixY = minY + iY
            ixZ = minZ + iZ
            stirvol[ixZ, ixY, ixX] = val
    
    return stirvol

def Normalization(reconspace, backprojector, sinogram):
    sinseg = sinogram.get_segment_by_sinogram(0)
    filler = np.ones(np.shape(stirextra.to_numpy(sinseg)))
    fillStirSpace(sinseg, filler)
    sinogram.set_segment(sinseg)

    backprojector.back_project(reconspace, sinogram)
    npReconSpace = stirextra.to_numpy(reconspace)
    pyvpx.numpy2vpx(npReconSpace, 'NormMap.vpx')
    return reconspace

def MLEMRecon(normMap, reconspace, imspaceError, forwardprojector, backprojector, sinogram, measurement, nIter = 30):
    imspaceError.fill(1)
    
    meascomp = measurement.get_segment_by_sinogram(0)
    npNormMap = stirextra.to_numpy(normMap)

    for iter in range(nIter):
        sinogram.fill(0)
        forwardprojector.forward_project(sinogram, reconspace);
        sinocomp = sinogram.get_segment_by_sinogram(0)
    
        #populate error sinogram
        error = stirextra.to_numpy(meascomp) /stirextra.to_numpy(sinocomp)
        error[np.isnan(error)] = 1
        error[np.isinf(error)] = 1
        #error[error > 1E10] = 1;
        #error[error < 1E-10] = 1;
        
        sinocompy = fillStirSpace(sinocomp, error)
        sinogram.set_segment(sinocompy)

        #backproject error
        backprojector.back_project(imspaceError, sinogram)
        image = stirextra.to_numpy(reconspace)
        error = stirextra.to_numpy(imspaceError)
        
        error = error/npNormMap
        #error[error > 1E10] = 0
        error[np.isinf(error)] = 0
        error[np.isnan(error)] = 1
        #pylab.imshow(error[0,:,:]), pylab.show()

        #update image
        image = image * error
        reconspace = fillStirSpace(reconspace, image)
    return reconspace

def forwardProjectionShiftError(yShift, forwardProjector, shiftingSino, referenceSino, image, shiftImage, iter, iFrame):
    
    npimage = stirextra.to_numpy(image)

    npshiftim = np.zeros(np.shape(npimage))
    ndimage.shift(npimage, [0, yShift, 0], output = npshiftim)
    shiftImage = fillStirSpace(shiftImage, npshiftim)
    
    #Forward project
    forwardProjector.forward_project(shiftingSino, shiftImage)

    #Calculate error measure
    shiftingSinoExtr =  stirextra.to_numpy(shiftingSino.get_segment_by_sinogram(0))
    shiftingSinoExtr /= np.sum(shiftingSinoExtr)

    referenceSinoExtr = stirextra.to_numpy(referenceSino.get_segment_by_sinogram(0))
    referenceSinoExtr /= np.sum(referenceSinoExtr)

    npShiftingSino =shiftingSinoExtr - referenceSinoExtr

    npShiftingSino[np.isnan(npShiftingSino)]   = 0
    npShiftingSino[np.isinf(npShiftingSino)]   = 0

    return np.sum(npShiftingSino**2)

def OptimizeSinogramError(vShift, forwardProjector, sinogramList, measurementList, image, shiftImage, SurrogateSignal):
    iDim1 = 0
    localShift = vShift
    def calcError():
        error = 0
        modLocalShift = localShift[0] * SurrogateSignal + localShift[1]
        #print(modLocalShift)
        for iFrame in range(nFrames):
            error += forwardProjectionShiftError(modLocalShift[iFrame], forwardProjector, sinogramList[iFrame], measurementList[iFrame], image, shiftImage, iDim1, iFrame)
        return error

    nFrames = len(SurrogateSignal)
    #shift image
    yShift = vShift[0] * SurrogateSignal + vShift[1]
    
    #get start error
    startError = calcError()
    currentError = startError
    errorLog = []    

    tries = 0
    refDims = np.zeros(2)
   # while tries < 10:
   #     tries += 1
    iDim2 = 0
    for iDim1 in range(16):
        localShift = np.array([ vShift[0] + iDim1,  vShift[1] + iDim2])
        currentError = calcError()
        errorLog.append(currentError)
        if currentError < startError:
            startError = currentError
            
            refDims = [iDim1, iDim2]
    
    return [refDims[0], refDims[1]], errorLog

                
