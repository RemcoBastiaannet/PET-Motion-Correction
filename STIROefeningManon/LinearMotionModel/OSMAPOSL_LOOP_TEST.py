import stir
import stirextra
import pylab
import numpy as np 
import os
from StirSupport import *
import matplotlib.pyplot as plt

nRings = 1
span = 1 
max_ring_diff = 0 

scanner = stir.Scanner(stir.Scanner.Siemens_mMR)
scanner.set_num_rings(nRings)

projdata_info = stir.ProjDataInfo.ProjDataInfoCTI(scanner, span, max_ring_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins(), False)

guessP = np.ones((1, 260, 260))

guessS = stir.FloatVoxelsOnCartesianGrid(projdata_info, 1,
                    stir.FloatCartesianCoordinate3D(stir.make_FloatCoordinate(0,0,0)),
                    stir.IntCartesianCoordinate3D(stir.make_IntCoordinate(1, 260, 260))) 

fillStirSpace(guessS, guessP) 

recon = stir.OSMAPOSLReconstruction3DFloat('config_TEST.par')

poissonobj = recon.get_objective_function()
poissonobj.set_recompute_sensitivity(True)

target = guessS

recon.set_up(target);
# Zonder attenuatie en scatter verandert de sensitiviteit (normalisatiemap) niet per iteratie, hij hangt nu alleen van dingen af als de scanner, dus die hoef je maar ��n keer te berekenen. 
poissonobj.set_recompute_sensitivity(False) 
for iter in range(1,10):
    recon.reconstruct(target);

    npimage = stirextra.to_numpy(target);
    plt.imshow(npimage[0,:,:]), plt.title('Iteration {}'.format(iter))
    plt.savefig('./Plaatjes/OSMAPOSL_LOOP_TEST/recon_{}.png'.format(iter))
    plt.show()