import sys
import stir
import stirextra
import pylab
import numpy as np
import os
import time
from StirSupport import *
from scipy.optimize import minimize

import matplotlib.pyplot as plt


nVoxelsXY = 256
nRings = 1
nLOR = 10
nFrames = 2
p = np.zeros((1,128, 128))

#Now we setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)
span = 1
max_ring_diff = 0

#Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)

#Setup Recon space
phantomspace      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                        stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                        stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(p)[0],np.shape(p)[1],np.shape(p)[2] ))) 

reconspace  = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(p)[0],np.shape(p)[1],np.shape(p)[2] ))) 

imspaceError = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(p)[0],np.shape(p)[1],np.shape(p)[2] ))) 

#Initialize the projection matrix (using ray-tracing)
MotionModel = stir.MotionModel()
projmatrix = stir.ProjMatrixByBinUsingRayTracing(MotionModel)

projmatrix.set_num_tangential_LORs(nLOR)
projmatrix.set_up(projdata_info, reconspace)

#Create projectors
forwardprojector    = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)


#Project - Fill recon space with phantom
phantomList     = []
measurementList = []
sinogramList    = []
reconList       = []

p = np.zeros((1,128, 128))
p[0, 50:60, 50:60] = 1

MotionModel.setOffset(0.0)
#Setup phantom
for iFrame in range(nFrames):

    if (iFrame > 0):
        MotionModel.setOffset(10.0)

    phantnp = np.zeros((1,128, 128))
    phantnp = p
    #offset = 6*iFrame
    #ndimage.shift(p, [0, offset, 0], output = phantnp)
    #pyvpx.numpy2vpx(phantnp, 'BasePhant.vpx')

    tmpphantspace = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                        stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                        stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(p)[0],np.shape(p)[1],np.shape(p)[2] ))) 

    tmpphantspace.fill(0)
    phant = fillStirSpace(tmpphantspace, phantnp)
    phantomList.append(phant)

    #allocate imagespace guestimates
    guestimatespace = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                        stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                        stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(p)[0],np.shape(p)[1],np.shape(p)[2] ))) 

    guestimatespace.fill(1)
    reconList.append(guestimatespace)
    # We'll just create the data in memory here
    measurementList.append(stir.ProjDataInMemory(stir.ExamInfo(), projdata_info))
    #forward project last measurement in list

    forwardprojector.forward_project(measurementList[-1], phantomList[-1]);
   
    tmp = measurementList[-1].get_segment_by_sinogram(0)
    #pyvpx.numpy2vpx(stirextra.to_numpy(tmp), ('MeasuredSinoFrame' + str(iFrame) + '.vpx'))
    #append empty sinogram for estimations
    sinogramList.append(stir.ProjDataInMemory(stir.ExamInfo(), projdata_info))


SurrogateSignal = np.array(range(nFrames))

#Normalize
NormSpace = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(p)[0],np.shape(p)[1],np.shape(p)[2] ))) 

NormSpace = Normalization(NormSpace, backprojector, stir.ProjDataInMemory(stir.ExamInfo(), projdata_info))

locationList= []
#sinosum = np.zeros((1,252,344))
for iFrame in range(nFrames):
    reconList[iFrame] = MLEMRecon(NormSpace, reconList[iFrame], imspaceError, forwardprojector, backprojector, sinogramList[iFrame], measurementList[iFrame], nIter = 5)
    #sinosum += stirextra.to_numpy(measurementList[iFrame].get_segment_by_sinogram(0))
recon = pyvpx.numpy2vpx( stirextra.to_numpy(reconList[0]), 'Recon0.vpx')
recon = pyvpx.numpy2vpx( stirextra.to_numpy(reconList[1]), 'Recon1.vpx')
recon = pyvpx.numpy2vpx( stirextra.to_numpy(reconList[1]) - stirextra.to_numpy(reconList[0]) , 'ReconDiff.vpx')
#recnp = stirextra.to_numpy(reconList[0])
#plt.imshow(recnp[0,:,:]), plt.show()


'''
guestimatespace = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(p)[0],np.shape(p)[1],np.shape(p)[2] ))) 

vShift = np.array([0,0,0])
vShiftOut, errorLog = OptimizeSinogramError(vShift, forwardprojector,sinogramList ,measurementList ,reconList[0], guestimatespace,  SurrogateSignal)
print(vShiftOut)

plt.figure(1)
plt.plot(errorLog)

base = stirextra.to_numpy(reconList[0])
ref = stirextra.to_numpy(reconList[3])
shifty = stirextra.to_numpy(reconList[2])

plt.figure(2)
plt.imshow(base[0,:,:])

plt.figure(3)
plt.imshow(ref[0,:,:])

ndimage.shift(base, [0, vShiftOut[0]*3, 0], output = shifty)
plt.figure(4)
plt.imshow(shifty[0,:,:] - ref[0,:,:])

plt.show()


#recnp = stirextra.to_numpy(reconList[0])
#plt.imshow(recnp[0,:,:]), plt.show()
'''