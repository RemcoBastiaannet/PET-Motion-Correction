import sys
import stir
import stirextra
import pylab
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from StirSupport import *
from scipy.optimize import minimize
from prompt_toolkit import input


nVoxelsXY = 256
nRings = 1
nLOR = 10
nFrames = 2
span = 1 # No axial compression  
max_ring_diff = 0 # maximum ring difference between the rings of oblique LORs 
trueShiftPixels = 40; # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd)  


# Setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)

# Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)


#_______________________PHANTOM______________________________________
# Python 
phantomP = [] 
for iFrame in range(nFrames): 
    tmp = np.zeros((1, 128, 128)) 
    tmp[0, (10+iFrame*trueShiftPixels):(30+iFrame*trueShiftPixels), 60:80] = 1
    phantomP.append(tmp) 
originalImageP = phantomP[0]

plt.figure(1)
plt.subplot(1,2,1), plt.title('Phantom TF 1'), plt.imshow(phantomP[0][0,:,:]) 
plt.subplot(1,2,2), plt.title('Phantom TF 2'), plt.imshow(phantomP[1][0,:,:]) 
plt.show()

# STIR 
originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(originalImageS, originalImageP)

phantomS = []
imageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
for iFrame in range(nFrames): 
    fillStirSpace(imageS, phantomP[iFrame])
    phantomS.append(imageS)


#_______________________PROJ MATRIX AND PROJECTORS______________________
slope = 0.0 
offSet = 0.0 
MotionModel = stir.MotionModel(nFrames, slope, offSet) # A motion model is compulsory  
projmatrix = stir.ProjMatrixByBinUsingRayTracing(MotionModel)
projmatrix.set_num_tangential_LORs(nLOR)
projmatrix.set_up(projdata_info, originalImageS)

# Create projectors
forwardprojector    = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)


#_________________________MEASUREMENT_______________________________
measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
measurementListP = [] 

## First time frame 
forwardprojector.forward_project(measurement, phantomS[0])
measurement.write_to_file('sinoMeas_1.hs')
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)
measurementListP.append(measurementP) 

## Second time frame 
forwardprojector.forward_project(measurement, phantomS[1])
measurement.write_to_file('sinoMeas_2.hs')
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)
measurementListP.append(measurementP) 


#_________________________PROJECTIONS OF GUESS __________________
'''Negeer voor nu het initial estimate, reconstrueer in één keer het blokje en focus op het vinden van de beweging van dit blokje'''
projection = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)

reconImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
reconImageS.fill(1) # moet er staan 

MotionModel.setOffset(0.0)
reconOSMAPOSL = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_1.par')
reconOSMAPOSL.set_up(reconImageS)
recon = reconOSMAPOSL.reconstruct(reconImageS)
guessS = recon
guessP = stirextra.to_numpy(reconImageS)
plt.imshow(guessP[0,:,:]), plt.title('Initial guess'), plt.show() 

## Second time frame
quadErrorSumList = [] 
for offset in range(0, -60, -10):
    MotionModel.setOffset(-offset) 
    forwardprojector.forward_project(projection, guessS)
    projection.write_to_file('sinoGuess_1.hs')
    projectionS = projection.get_segment_by_sinogram(0)
    projectionP = stirextra.to_numpy(projectionS)

    plt.figure(2) 
    plt.subplot(1,2,1), plt.imshow(projectionP[0,:,:]), plt.title('Current proj TF2')
    plt.subplot(1,2,2), plt.imshow(measurementListP[1][0,:,:]), plt.title('Measurement TF2')
    plt.show() 

    # Dit gaat nog niet goed 
    # Je moet de combinatie van de projectie van TF1 en de projectie van TF2 vergelijken met de meting
    # Nu wordt alleen de projectie van TF2 gebruikt 
    quadErrorSum = np.sum((projectionP[0,:,:] - measurementListP[1][0,:,:])**2)
    quadErrorSumList.append(quadErrorSum)
    if quadErrorSum < 50: 
        print 'Motion shift was found to be:', offset
        break


















#_________________________FIRST RECONSTRUCTION________________________
# Measurement/projections of inital time frame
measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
forwardprojector.forward_project(measurement, guessS)
measurement.write_to_file('sino_1.hs')
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)

# Image reconstruction using OSMAPOSL 
reconImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
reconImageS.fill(1) # moet er staan 

MotionModel.setOffset(0.0)
reconOSMAPOSL = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_1.par')
s = reconOSMAPOSL.set_up(reconImageS)
reconOSMAPOSL.reconstruct(reconImageS)
reconImagePRef = stirextra.to_numpy(reconImageS) # reference time frame


#_________________________SECOND RECONSTRUCTION________________________
# Measurement/projections of inital time frame
measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
forwardprojector.forward_project(measurement, guessS)
measurement.write_to_file('sino_2.hs') 
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)

reconImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

quadErrorSumList = [] 
reconImagePList = []
for offset in range(0, -60, -10): 
    # Image reconstruction using OSMAPOSL 
    reconImageS.fill(1) # moet er staan 
    MotionModel.setOffset(offset)
    reconOSMAPOSL = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_2.par')
    reconOSMAPOSL.set_up(reconImageS)
    reconOSMAPOSL.reconstruct(reconImageS) 
    
    reconImageP = stirextra.to_numpy(reconImageS)
    reconImagePDict = {'recon' : reconImageP, 'offset' : offset}
    reconImagePList.append(reconImagePDict)

    quadErrorSum = np.sum((reconImageP[0,:,:] - reconImagePRef[0,:,:])**2)
    if quadErrorSum < 50: 
        print 'Motion shift was found to be:', offset
        break; 

plt.figure(2)
for i in range(len(reconImagePList)): 
    plt.subplot(2,3,i+1), plt.imshow(reconImagePList[i]['recon'][0,:,:]), plt.title('TF2 offset: ' + str(reconImagePList[i]['offset'])) 
plt.show()

plt.figure(3) 
plt.subplot(1,3,1), plt.imshow(reconImagePRef[0,:,:]), plt.title('OSMAPOSL recon TF 1')
plt.subplot(1,3,2), plt.imshow(phantomP[1][0,:,:]), plt.title('Phantom TF2')
plt.subplot(1,3,3), plt.imshow(reconImageP[0,:,:]), plt.title('OSMAPOSL recon TF 2 MC')
plt.show()


#_________________________COMBINING RECONSTRUCTIONS________________________
reconImagePCombined = [0.5*sum(x) for x in zip(reconImageP[0,:,:], reconImagePRef[0,:,:])]
plt.imshow(reconImagePCombined), plt.title('Combined reconstructions (with motion correction)'), plt.show() 