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

showImages = True   

nVoxelsXY = 256
nRings = 1
nLOR = 10
nFrames = 2

# Setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)
span = 1 # No axial compression  
max_ring_diff = 0 # maximum ring difference between the rings of oblique LORs 
trueShiftPixels = 40; # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd)  

# Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)

# Phantoms for each time frame
phantomP = [] 

# Create the individual time frames, the phantom is shifted in each frame w.r.t. the previous one 
for iFrame in range(nFrames): 
    tmp = np.zeros((1, 128, 128)) 
    tmp[0, (10+iFrame*trueShiftPixels):(30+iFrame*trueShiftPixels), 60:80] = 1
    phantomP.append(tmp) 

originalImageP = phantomP[0]
originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(originalImageS, originalImageP)

plt.figure(1)
plt.subplot(1,2,1), plt.title('Phantom TF 1'), plt.imshow(phantomP[0][0,:,:]) 
plt.subplot(1,2,2), plt.title('Phantom TF 2'), plt.imshow(phantomP[1][0,:,:]) 
plt.show()

phantomS = []
for iFrame in range(nFrames): 
    imageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
    fillStirSpace(imageS, phantomP[iFrame])
    phantomS.append(imageS)


# Initialize the projection matrix (using ray-tracing) 
slope = 0.0 
offSet = 0.0 # Do not shift the first projection (reference frame)  
MotionModel = stir.MotionModel(nFrames, slope, offSet) # A motion model is compulsory  
projmatrix = stir.ProjMatrixByBinUsingRayTracing(MotionModel)
projmatrix.set_num_tangential_LORs(nLOR)
projmatrix.set_up(projdata_info, originalImageS)

# Create projectors
forwardprojector    = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)


#_________________________FIRST RECONSTRUCTION________________________
# Measurement/projections of inital time frame
measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
forwardprojector.forward_project(measurement, phantomS[0])
measurement.write_to_file('sino_1.hs')
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)
#plt.imshow(measurementP[0,:,:]), plt.title('Sinogram time frame 1'), plt.show()

# Image reconstruction using OSMAPOSL 
reconImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
reconImageS.fill(1) # moet er staan 

MotionModel.setOffset(0.0)
reconOSMAPOSL = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_1.par')
s = reconOSMAPOSL.set_up(reconImageS)
reconOSMAPOSL.reconstruct(reconImageS)
reconImagePRef = stirextra.to_numpy(reconImageS) # reference time frame
reconImagePList.append(reconImagePRef)
#plt.imshow(reconImageP[0,:,:]), plt.title('OSMAPOSL reconstruction time frame 1'), plt.show()


measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
forwardprojector.forward_project(measurement, phantomS[1])

quadErrorSumList = [] 
#### PROBLEEM: als je de code meerdere keren uitvoert doet het raar 
for par1 in range(0, -60, -10): 
    #_________________________SECOND RECONSTRUCTION________________________
    # Measurement/projections of inital time frame
    measurement.write_to_file('sino_2.hs') # Hier gaat het mogelijk mis, als deze niet wordt overschreven maar toegevoegd bijvoorbeeld.
    measurementS = measurement.get_segment_by_sinogram(0)
    measurementP = stirextra.to_numpy(measurementS)
    #plt.imshow(measurementP[0,:,:]), plt.title('Sinogram time frame 2'), plt.show()

    # Image reconstruction using OSMAPOSL 
    reconImageS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                        stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                        stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
    reconImageS.fill(1) # moet er staan 

    MotionModel.setOffset(par1)
    reconOSMAPOSL = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_2.par')
    s = reconOSMAPOSL.set_up(reconImageS)
    reconOSMAPOSL.reconstruct(reconImageS)
    reconImageP = stirextra.to_numpy(reconImageS)

    quadErrorSum = np.sum((reconImageP[0,:,:] - reconImagePRef[0,:,:])**2)
        
    if quadErrorSum < 50: 
        print 'Motion shift was found to be:', par1
        break; 

plt.figure(2) 
plt.subplot(1,3,1), plt.imshow(reconImagePRef[0,:,:]), plt.title('OSMAPOSL recon TF 1')
plt.subplot(1,3,2), plt.imshow(phantomP[1][0,:,:]), plt.title('Phantom TF2')
plt.subplot(1,3,3), plt.imshow(reconImageP[0,:,:]), plt.title('OSMAPOSL recon TF 2 MC')
plt.show()

#_________________________COMBINING RECONSTRUCTIONS________________________
reconImagePCombined = [0.5*sum(x) for x in zip(reconImageP[0,:,:], reconImagePRef[0,:,:])]
plt.imshow(reconImagePCombined), plt.title('Combined reconstructions (with motion correction)'), plt.show() 