import numpy as np
import matplotlib.pyplot as plt
import ManonsFunctions as mf

stationary = False 
nGates = 5

filePath = "E:/Manon/Resultaten_Simulaties/4_Coregistratie/"

if (stationary): filePath += "Stationair/"
else: filePath += "Niet-stationair/"

gatedImagesReg = []
for iGate in range(1, nGates+1):
    gatedImagesReg.append(mf.load_itk(filePath + "Coregistered_gates/Gate{}.mhd".format(iGate, iGate)))
meanImageRegistered = np.mean(gatedImagesReg, axis = 0)
plt.figure(), plt.title('Mean with registration'), plt.imshow(meanImageRegistered[:,:,0], interpolation = None, vmin = 0, vmax = np.max(gatedImagesReg[0]), cmap=plt.cm.Greys_r)
plt.savefig(filePath + "MeanWithRegistration.png")
plt.close()

mf.writeMhdFile(meanImageRegistered, filePath + "MeanRegistered.mhd")