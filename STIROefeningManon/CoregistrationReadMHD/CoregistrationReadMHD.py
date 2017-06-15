import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import ManonsFunctionsHysteresis as mf

filePath = "E:/Manon/Resultaten_Simulaties/4_Coregistratie/Stationair/"

nGates = 5
gatedImages = []
for iGate in range(1, nGates+1):
    gatedImages.append(mf.load_itk(filePath + "Seperate_gates_unregistered/Gate{}.mhd".format(iGate)))
meanImageUnregistered = np.mean(gatedImages, axis = 0)
plt.figure(), plt.title('Mean without registration'), plt.imshow(meanImageUnregistered[:,:,0], interpolation = None, vmin = 0, vmax = np.max(gatedImages[0]), cmap=plt.cm.Greys_r)
plt.savefig(filePath + "MeanWithoutRegistration.png")
plt.close()

meanImageRegistered = load_itk(filePath + "Slicer_output_registered/RegisteredVolume.mhd")
meanImageRegistered /= 5
plt.figure(), plt.title('Mean with registration'), plt.imshow(meanImageRegistered[:,:,0], interpolation = None, vmin = 0, vmax = np.max(gatedImages[0]), cmap=plt.cm.Greys_r)
plt.savefig(filePath + "MeanWithRegistration.png")
plt.close()

mf.writeMhdFile(meanImageUnregistered, filePath + "MeanUnregistered.mhd")
mf.writeMhdFile(meanImageRegistered, filePath + "MeanRegistered.mhd")