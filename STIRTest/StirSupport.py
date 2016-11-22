import sys
import stir
import stirextra
import pylab
import numpy as np
import os
import time
import scipy as sp
import scipy.ndimage as ndimage


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

def MLEMRecon(reconspace, imspaceError, forwardprojector, backprojector, sinogram, measurement, nIter = 30):
    imspaceError.fill(1)
    
    meascomp = measurement.get_segment_by_sinogram(0)

    for iter in range(nIter):
        forwardprojector.forward_project(sinogram, reconspace);
        sinocomp = sinogram.get_segment_by_sinogram(0)
    
        #populate error sinogram
        error = stirextra.to_numpy(meascomp) /stirextra.to_numpy(sinocomp)
        error[np.isnan(error)] = 0
        error[np.isinf(error)] = 0
        error[error > 1E10] = 0;
        error[error < 1E-10] = 0;

        sinocomp = fillStirSpace(sinocomp, error)
        sinogram.set_segment(sinocomp)

        #backproject error
        backprojector.back_project(imspaceError, sinogram)
        image = stirextra.to_numpy(reconspace)
        error = stirextra.to_numpy(imspaceError)
        #pylab.imshow(error[0,:,:]), pylab.show()

        #update image
        image = image * error
        reconspace = fillStirSpace(reconspace, image)
    return reconspace

def forwardProjectionShiftError(vShift, forwardProjector, shiftingSino, referenceSino, image):
    #shift image
    npimage = stirextra.to_numpy(image)
    ndimage.shift(npimage, vShift, output = npimage)
    image = fillStirSpace(image, npimage)
    
    #Forward project
    forwardProjector.forward_project(shiftingSino, image)

    #Calculate error measure
    shiftingSinoExtr = shiftingSino.get_segment_by_sinogram(0)
    referenceSinoExtr = referenceSino.get_segment_by_sinogram(0)
    npShiftingSino = stirextra.to_numpy(shiftingSinoExtr) / stirextra.to_numpy(referenceSinoExtr)
    
    npShiftingSino[np.isnan(npShiftingSino)]   = 0
    npShiftingSino[np.isinf(npShiftingSino)]   = 0

    npShiftingSino[npShiftingSino > 1E6]    = 0
    npShiftingSino[npShiftingSino < 1E-6]   = 0

    return np.sum(npShiftingSino)