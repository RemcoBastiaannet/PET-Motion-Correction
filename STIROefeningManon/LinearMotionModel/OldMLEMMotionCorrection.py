'''
HIER HOREN NOG WAT DINGEN BOVEN TE STAAN
ZIE LinearMotionModel1DImageSpace 
'''

#______________________________OUD - motion correction tijdens MLEM, normalisatie werkt niet goed____________________________

# Create projectors
forwardprojector    = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector       = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)

### Measurement/projections of the inital time frames 
measurementPhantomPlist = []
for iFrame in range(nFrames): 
    measurement = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    forwardprojector.forward_project(measurement, phantomS[iFrame]);  
    measurementS = measurement.get_segment_by_sinogram(0)
    measurementP = stirextra.to_numpy(measurementS)
    measurementPhantomPlist.append(measurementP) 
### 

# MLEM 
# Initial guess 
guessImageP = np.ones(np.shape(originalImageP)) # Dit moet waarschijnlijk niet het eerste plaatje zijn. 
guessImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 
guessImageSlist = []
guessImagePlist = []
guessImagePlist.append(guessImageP) 
guessSinogramPlist = []
errorPTotal = 1 

# TEST 
fillStirSpace(guessImageS, guessImageP)
normSinogram = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
forwardprojector.forward_project(normSinogram, guessImageS)
tmp = normSinogram.get_segment_by_sinogram(0)
normSinogramP = stirextra.to_numpy(tmp)
plt.imshow(normSinogramP[0,:,:]), plt.show()
fillStirSpace(tmp, normSinogramP)
normSinogram.set_segment(tmp)
 
backprojector.back_project(guessImageS, normSinogram)

b = stirextra.to_numpy(guessImageS)
plt.imshow(b[0,:,:]), plt.show()
# EINDE TEST 

for i in range(nMLEM): 
    par1 = -10 # Dit is de juiste shift 

    # update current guess 
    fillStirSpace(guessImageS, guessImageP)
    guessImageSlist.append(guessImageS)
 
    # Forward project initial guess  
    for iFrame in range(nFrames): 
        guessSinogram = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
        MotionModel.setOffset(par1*iFrame) # Ieder frame heeft zijn eigen shift (als er meer tijd is verstreken, is de shift groter), hoe groot precies hangt af van een parameter
        forwardprojector.forward_project(guessSinogram, guessImageS)
        guessSinogramS = guessSinogram.get_segment_by_sinogram(0)
        guessSinogramP = stirextra.to_numpy(guessSinogramS)
        guessSinogramPlist.append(guessSinogramP)
        errorP = measurementPhantomPlist[iFrame]/guessSinogramP
        errorP[np.isnan(errorP)] = 0
        errorP[np.isinf(errorP)] = 0
        errorP[errorP > 1E10] = 0
        errorP[errorP < 1E-10] = 0
        plt.figure(3), plt.imshow(errorP[0,:,:]), plt.title('Sinogram error'), plt.show() 

        fillStirSpace(guessSinogramS, errorP)
        guessSinogram.set_segment(guessSinogramS)

        errorBackprS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                        stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                        stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

        MotionModel.setOffset(par1*iFrame)
        backprojector.back_project(errorBackprS, guessSinogram)
        errorBackprP = stirextra.to_numpy(errorBackprS) 
        plt.figure(4), plt.imshow(errorBackprP[0,:,:]), plt.title('Backprojection sinogram error'), plt.show() 
        errorPTotal *= errorBackprP
    #plt.figure(5), plt.imshow(errorPTotal[0,:,:]), plt.title('Total error'), plt.show() 
        #plt.figure(3), plt.subplot(1,3,iFrame+1), plt.imshow(guessSinogramP[0,:,:])
    #plt.show() 

    # Normalization - werkt nog niet correct! 
    normalizationSinogramP = np.ones(np.shape(measurementP)) 
    normalizationSinogramS = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info)
    normalizationSinogram = normalizationSinogramS.get_segment_by_sinogram(0)
    fillStirSpace(normalizationSinogram, normalizationSinogramP) 
    normalizationSinogramS.set_segment(normalizationSinogram)

    normalizationS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(np.shape(originalImageP)[0],np.shape(originalImageP)[1],np.shape(originalImageP)[2] ))) 

    MotionModel.setOffset(par1*iFrame) # normalisatie moet frame specifiek worden 
    backprojector.back_project(normalizationS, normalizationSinogramS)
    normalizationP = stirextra.to_numpy(normalizationS)
    plt.figure(5), plt.imshow(normalizationP[0,:,:]), plt.title('Normalization'), plt.show() 

    # Update guess 
    guessImageP = stirextra.to_numpy(guessImageS)
    errorBackprP = stirextra.to_numpy(errorBackprS)
    norm = 1/(3*b)
    norm[np.isnan(norm)] = 0
    norm[np.isinf(norm)] = 0
    plt.figure(3), plt.imshow(norm[0,:,:]), plt.show()
    guessImageP *= errorPTotal
    guessImagePlist.append(guessImageP) # voor visualisatie, guessImageP heeft nu de laatste, mocht je die nodig hebben
 
if (showImages): 
    plt.figure(5)
    plt.imshow(guessImagePlist[1][0,:,:])
    plt.show()