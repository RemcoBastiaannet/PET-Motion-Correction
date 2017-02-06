import sys
import stir
import stirextra
import pylab
import numpy as np
import math 
import os
import time
import matplotlib.pyplot as plt
from StirSupport import *
from scipy.optimize import minimize
from prompt_toolkit import input

import scipy as sp
from skimage.io import imread
from skimage import data_dir
from skimage.transform import iradon, radon, rescale

plt.ioff() # Turn interactive plotting off 

nRings = 1
span = 1 
max_ring_diff = 0 
nLOR = 10 

scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)

projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)

image = np.zeros((1,160,160))
image[0, 65:95, 65:95] = 1 

originalImageS      = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(1, 160, 160)))  
fillStirSpace(originalImageS, image)

guessP = np.ones((1, 160, 160)) # Voor het blokje 

guessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(1, 160, 160))) 

fillStirSpace(guessS, guessP) 

slope = 0.0 
offSet = 0.0 
nFrames = 1 
MotionModel = stir.MotionModel(nFrames, slope, offSet) # A motion model is compulsory  
projmatrix = stir.ProjMatrixByBinUsingRayTracing(MotionModel)
projmatrix.set_num_tangential_LORs(nLOR)
projmatrix.set_up(projdata_info, originalImageS)

recon = stir.OSMAPOSLReconstruction3DFloat(projmatrix, 'config_TEST.par') # Uses sinoMeas_1.hs from LinearMotionModel1DProjSpace.py 
poissonobj = recon.get_objective_function()
poissonobj.set_recompute_sensitivity(True)

target = guessS

recon.set_up(target);
# Zonder attenuatie en scatter verandert de sensitiviteit (normalisatiemap) niet per iteratie, hij hangt nu alleen van dingen af als de scanner, dus die hoef je maar ��n keer te berekenen. 
poissonobj.set_recompute_sensitivity(False) 

# Je moet er nog steeds wel voor zorgen dat het mapje bestaat! 
num_subsets = recon.get_num_subsets() 
num_iterations = recon.get_num_subiterations() 

for iter in range(1,5):
    MotionModel.setOffset(20.0)
    recon.reconstruct(target)

    npimage = stirextra.to_numpy(target);
    plt.imshow(npimage[0,:,:], cmap=plt.cm.Greys_r, interpolation=None, vmin = 0), plt.title('Iteration {}'.format(iter))
    plt.savefig('./Figures/Tests/OSMAPOSL_loop_test/{}_sub_{}_it/recon_{}.png'.format(num_subsets, num_iterations, iter))
    plt.show()