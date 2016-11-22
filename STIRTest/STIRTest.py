import sys
import stir
import stirextra
import pylab
import numpy as np
import os
import time
from StirSupport import *
from scipy.optimize import minimize


nVoxelsXY = 256
nRings = 1
nLOR = 10
nFrames = 15
p = np.zeros((1,128, 128))

#Now we setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)
span = 1
max_ring_diff = 0

#Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)

#Setup Recon space
target      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(p)[0],np.shape(p)[1],np.shape(p)[2] ))) 

reconspace  = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(p)[0],np.shape(p)[1],np.shape(p)[2] ))) 

imspaceError = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(p)[0],np.shape(p)[1],np.shape(p)[2] ))) 

#Initialize the projection matrix (using ray-tracing)
projmatrix = stir.ProjMatrixByBinUsingRayTracing()
projmatrix.set_num_tangential_LORs(nLOR)
projmatrix.set_up(projdata_info, target)

#Create projectors
forwardprojector    = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)


#Project - Fill recon space with phantom
phantomList     = []
measurementList = []
sinogramList    = []
reconList       = []
#Setup phantom
for iFrame in range(nFrames):
    p = np.zeros((1,128, 128))
    offset = 10*iFrame
    p[0,offset:offset+10, 50:60] = 1
    target.fill(0)
    phantomList.append(fillStirSpace(target, p))
    #allocate imagespace guestimates
    target.fill(1)
    reconList.append(target)
    # We'll just create the data in memory here
    measurementList.append(stir.ProjDataInMemory(stir.ExamInfo(), projdata_info))
    #forward project last measurement in list
    forwardprojector.forward_project(measurementList[-1], phantomList[-1]);
    #append empty sinogram for estimations
    sinogramList.append(stir.ProjDataInMemory(stir.ExamInfo(), projdata_info))

locationList= []
for iTer in range(100):
    for iFrame in range(nFrames):
        reconList[iFrame] = MLEMRecon(reconList[0], imspaceError, forwardprojector, backprojector, sinogramList[iFrame], measurementList[iFrame], nIter = 30)
        forwardprojector.forward_project(sinogramList[iFrame], reconList[iFrame]);        
        vShift = [0,0,0]
        res = minimize(forwardProjectionShiftError, vShift, args=(forwardprojector, sinogramList[iFrame], measurementList[iFrame], reconList[iFrame]), options={'maxiter':5})
        locationList.append(res)
    

reconspace.fill(1)
#


#pylab.imshow(image[0,:,:]), pylab.show()


