import stir
import stirextra
import numpy as np

##ADJUST THESE SETTINGS
#max_ring_diff = 49
max_ring_diff = 5
sizeOfVolumeInVoxels = (100,200,200)


#ready up the stir
scanner = stir.Scanner(stir.Scanner.E1104)
scanner.set_num_rings(55)
#Setup projection data
span = 11


#projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, 168, scanner.get_max_num_non_arccorrected_bins(), False)
projdata_info2D = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, 168, scanner.get_max_num_non_arccorrected_bins(), False)


guessVolume = stir.FloatVoxelsOnCartesianGrid(projdata_info2D, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(sizeOfVolumeInVoxels[0], sizeOfVolumeInVoxels[1], sizeOfVolumeInVoxels[2])))
guessVolume.fill(1)

ErrorVolume = stir.FloatVoxelsOnCartesianGrid(projdata_info2D, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(sizeOfVolumeInVoxels[0], sizeOfVolumeInVoxels[1], sizeOfVolumeInVoxels[2]))) 
ErrorVolume.fill(0)

forwardSino2D = stir.ProjDataInMemory(stir.ExamInfo(), projdata_info2D)
forwardSino2D.fill(0)

#Initialize the projection matrix (using ray-tracing)
projmatrix2D =  stir.ProjMatrixByBinUsingRayTracing()
nLOR = 10
projmatrix2D.set_num_tangential_LORs(nLOR)
projmatrix2D.set_up(projdata_info2D, guessVolume)

#Create projectors
forwardprojector2D = stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix2D)
backprojector2D = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix2D)

def forwardProject(npSource):
    guessVolume.fill(npSource.flat)
    forwardSino2D.fill(0)
    forwardprojector2D.forward_project(forwardSino2D, guessVolume)
    return stirextra.to_numpy(forwardSino2D)

def backProject(npSino):
    guessVolume.fill(0)
    forwardSino2D.fill(npSino.flat)
    backprojector2D.back_project(guessVolume, forwardSino2D)
    return stirextra.to_numpy(guessVolume)
