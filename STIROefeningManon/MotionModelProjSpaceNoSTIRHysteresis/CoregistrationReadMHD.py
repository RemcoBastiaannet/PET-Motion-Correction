import numpy as np
import matplotlib.pyplot as plt
import ManonsFunctions as mf
import pyvpx

nGates = 5
experiment = 'Sinus2D/'

filePath = "E:/Manon/Resultaten_Experimenten/" + experiment + "Gating/"

gatedImagesReg = []
for iGate in range(1, nGates+1):
    gatedImagesReg.append(mf.load_itk(filePath + "Coregistered_gates/Gate{}.mhd".format(iGate, iGate)))
meanImageRegistered = np.mean(gatedImagesReg, axis = 0)

pyvpx.numpy2vpx(meanImageRegistered, filePath + "MeanRegistered.vpx")
mf.writeMhdFile(meanImageRegistered, filePath + "MeanRegistered.mhd")