import numpy as np
from ctypes import *
from numpy.ctypeslib import ndpointer
from STIRComponents import *

class SinoReader:
    #HELP: 1) run constructor with or without specific path to dll; 2) set LM file; 3) update sinogram with begin time + duration; 4) object.Sinogram contains NumPy sinogram
    def setupDLL(self):
        self.fLM = None
        self.c_initLMFile = self.LMDLL.initFile
        self.c_initLMFile.argtypes = [c_char_p, ndpointer(c_int32, flags='CONTIGUOUS')] 
        self.c_initLMFile.restype = c_int

        self.c_getSino = self.LMDLL.getSino
        self.c_getSino.argtypes = [ndpointer(c_int32, flags='CONTIGUOUS'), c_int, c_int]
        self.c_getSino.restype = c_int;

    def __init__(self, fDLL = None):
        self.fLM = None
        if not fDLL:
            try:
                self.LMDLL = cdll.LoadLibrary(r'LMProc.dll')
                self.setupDLL()
            except WindowsError:
                print('Cannot find DLL on standard location. Please init this class with specific DLL location')
        else:
            self.LMDLL = cdll.LoadLibrary(fDLL)
            self.setupDLL()

    def setEmptySino(self, size = None):
        if size:
            self.Sino = np.zeros(size).astype(c_int32)
        else:
            self.Sino = np.zeros((621,168,400)).astype(c_int32)        

    def setLMFile(self, fLM):
        self.setEmptySino()
        self.fLM = fLM
        self.c_initLMFile(c_char_p(self.fLM), self.Sino)

    def updateSino(self, begin = 0, end = 5000000):
        self.c_getSino(self.Sino, c_int(begin), c_int(end))
        self.Sino = self.Sino[:109,:,:]

def generateNorm():
    print("NORM!")

'''
AA = SinoReader()
AA.setLMFile('pathtoLMFILE.ptd')

AA.updateSino(500, 500)
AA.Sino

npSino = forwardProject(npSource)
npSource = backProject(npSino)
'''
