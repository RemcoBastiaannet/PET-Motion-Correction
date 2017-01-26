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


#nVoxelsXY = 256 # ?
nRings = 1
nLOR = 10 # ? 
nFrames = 2
span = 1 # No axial compression  
max_ring_diff = 0 # maximum ring difference between the rings of oblique LORs 
trueShiftPixels = 6; # Kan niet alle waardes aannemen (niet alle shifts worden geprobeerd)  


# Setup the scanner
scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)


# Setup projection data
projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)


#_______________________PHANTOM______________________________________
# Create a block that is shifted from each time frame w.r.t. the previous one (constant shift)  
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

# Python -> STIR 
originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(originalImageS, originalImageP)

phantomS = []

image1S      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(image1S, phantomP[0])
phantomS.append(image1S)

image2S      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
fillStirSpace(image2S, phantomP[1])
phantomS.append(image2S)


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

# Projection of the first time frame 
MotionModel.setOffset(0.0)
forwardprojector.forward_project(measurement, phantomS[0])
measurement.write_to_file('sinoMeas_1.hs')
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)
measurementListP.append(measurementP) 

# Projection of the second time frame 
MotionModel.setOffset(0.0) # Beweging zit al in het plaatje 
forwardprojector.forward_project(measurement, phantomS[1])
measurement.write_to_file('sinoMeas_2.hs')
measurementS = measurement.get_segment_by_sinogram(0)
measurementP = stirextra.to_numpy(measurementS)
measurementListP.append(measurementP) 


#_________________________GUESS_______________________________
'''Negeer voor nu het initial estimate in die config files'''
projection = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)

# Reconstruction of the first time frame 
reconGuess1S = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  

MotionModel.setOffset(0.0)
reconGuess1S.fill(1) # moet er staan
recon1 = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_1.par')
recon1.set_up(reconGuess1S)
recon1.reconstruct(reconGuess1S)
guess1P = stirextra.to_numpy(reconGuess1S)

# Reconstruction of the second time frame 
reconGuess2S = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

MotionModel.setOffset(0.0)
reconGuess2S.fill(1) # moet er staan
recon2 = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_2.par')
recon2.set_up(reconGuess2S)
recon2.reconstruct(reconGuess2S)
guess2P = stirextra.to_numpy(reconGuess2S)

# Combining the two reconstructions in one initial guess by averaging them 
guessP = 0.5*(guess1P + guess2P)
plt.imshow(guessP[0,:,:]), plt.title('Initial guess'), plt.show() 

#_________________________COMBINED MODEL OPTIMIZATION, MOTION CORRECTION AND IMAGE RECONSTRUCTION_______________________________

for i in range(1): 
    guessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
    fillStirSpace(guessS, guessP)

    #_________________________MOTION MODEL OPTIMIZATION_______________________________
    quadErrorSumList = []

    # Try different offsets to see which one fits the measurements best 
    for offset in range(trueShiftPixels/2-1,trueShiftPixels/2+1,1): 
        projectionPList = []

        # Projection of the guess with shift +offset, this is our trial for the first time frame 
        MotionModel.setOffset(offset) 
        forwardprojector.forward_project(projection, guessS)
        projection.write_to_file('sino_1.hs')
        projectionS = projection.get_segment_by_sinogram(0)
        projectionP = stirextra.to_numpy(projectionS)
        projectionPList.append(projectionP)

        # Projection of the guess with shift -offset, this is our trial for the first time frame
        MotionModel.setOffset(-offset) 
        forwardprojector.forward_project(projection, guessS)
        projection.write_to_file('sino_2.hs')
        projectionS = projection.get_segment_by_sinogram(0)
        projectionP = stirextra.to_numpy(projectionS)
        projectionPList.append(projectionP)

        # Computing the quadratic error of these projections w.r.t. the measurements 
        quadErrorSum = np.sum((projectionPList[0][0,:,:] - measurementListP[0][0,:,:])**2) + np.sum((projectionPList[1][0,:,:] - measurementListP[1][0,:,:])**2)
        quadErrorSumList.append({'offset' : offset, 'quadErrorSum' : quadErrorSum})

    # Determine the offset for which the quadratic error is at a minimum 
    quadErrorSums = [x['quadErrorSum'] for x in quadErrorSumList]
    for i in range(len(quadErrorSumList)): 
        if(quadErrorSumList[i]['quadErrorSum'] == min(quadErrorSums)): 
            offsetFound = quadErrorSumList[i]['offset']


    #_________________________MOTION COMPENSATION_______________________________

    # Same code, but with the opposite of the offset that was actually found (correction) 
    projPListMC = [] 

    projUp = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    projDown = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)

    # Make new, corrected projection data for the first time frame
    MotionModel.setOffset(-10) 
    forwardprojector.forward_project(projDown, guessS)
    projDown.write_to_file('sino_MC_1.hs')
    projDownS = projection.get_segment_by_sinogram(0)
    projDownP = stirextra.to_numpy(projDownS)
    projPListMC.append(projDownP)

    # Make new, corrected projection data for the second time frame
    MotionModel.setOffset(+10) 
    forwardprojector.forward_project(projUp, guessS)
    projUp.write_to_file('sino_MC_2.hs')
    projUpS = projection.get_segment_by_sinogram(0)
    projUpP = stirextra.to_numpy(projUpS)
    projPListMC.append(projUpP)

    # Add the sinograms of first and second time frame 
    projSumMCP = np.add(projPListMC[0], projPListMC[1])
    fillStirSpace(projUpS, projSumMCP) 
    projUp.set_segment(projUpS) 
    projUp.write_to_file('sino_MC.hs') 


    #_________________________IMAGE RECONSTRUCTION_______________________________
    ### TEST 
    # OSMAPOSL reconstruction of the motion corrected and combined projection data 
    reconMCS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                        stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                        stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
    reconMCS.fill(1) # moet er staan

    recon = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_test1.par')
    recon.set_up(reconMCS)
    recon.reconstruct(reconMCS)

    reconMCP = stirextra.to_numpy(reconMCS) 
    plt.imshow(abs(guessP[0,:,:]-reconMCP[0,:,:])), plt.title('Test'), plt.show()

    # OSMAPOSL reconstruction of the motion corrected and combined projection data 
    reconMCS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                        stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                        stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
    reconMCS.fill(1) # moet er staan

    recon = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_test2.par')
    recon.set_up(reconMCS)
    recon.reconstruct(reconMCS)

    reconMCP = stirextra.to_numpy(reconMCS)
    plt.imshow(abs(guessP[0,:,:]-reconMCP[0,:,:])), plt.title('Test'), plt.show()
    ### 

    # OSMAPOSL reconstruction of the motion corrected and combined projection data 
    reconMCS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                        stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                        stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] )))  
    reconMCS.fill(1) # moet er staan

    recon = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_Proj_MC.par')
    recon.set_up(reconMCS)
    recon.reconstruct(reconMCS)

    reconMCP = stirextra.to_numpy(reconMCS)
    plt.imshow(reconMCP[0,:,:]), plt.title('Motion corrected image'), plt.show()

    #_________________________UPDATE GUESS_______________________________
    guessP = reconMCP 