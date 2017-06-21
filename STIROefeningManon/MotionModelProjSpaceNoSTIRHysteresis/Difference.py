import numpy as np
import matplotlib.pyplot as plt
import ManonsFunctionsHysteresis as mf

filePath = "E:/Manon/Resultaten_Simulaties/"

ref = mf.load_itk(filePath + "1_Geen_beweging_(referentie)/finalImage.mhd")

image = mf.load_itk(filePath + "3_Gaten/Stationair/Difference/Gate1.mhd")

print np.sum((ref - image)**2)