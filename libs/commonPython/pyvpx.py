import array
import sys
import numpy as np
from numpy import ma as ma
import scipy
from scipy import ndimage
import struct

def uint8touint32(n):
    tmp = n[0]
    tmp = tmp << 8
    tmp = tmp | n[1]
    tmp = tmp << 8
    tmp = tmp | n[2]
    tmp = tmp << 8
    tmp = tmp | n[3]
    return tmp

def uint32touint8(n):
    tmp = array.array("B", np.zeros(4, dtype=np.uint8))
    tmp[3] = n & 0xff
    tmp[2] = n>>8 & 0xff
    tmp[1] = n>>16 & 0xff
    tmp[0] = n>>24 & 0xff
    return tmp
    
def float32touint8(n):
    return array.array("B", struct.pack('>f', n))
    
    #fl = struct.pack('>f', n)
    #tmp = array.array("B", (ord(fl[0]), ord(fl[1]), ord(fl[2]), ord(fl[3])) )
    #return tmp
    
def vpx2numpy(vpxfile):
    # Read from file again
    b = array.array("B")
    with open(vpxfile, "rb") as f:
        b.fromfile(f, 100)
    if sys.byteorder == "little":
        b.byteswap()

        #header[24-27] =       X-dimension
    xdim = uint8touint32(b[24:28])
        #header[28-31] =       Y-dimension (optional)
    ydim = uint8touint32(b[28:32])
        #header[32-35] =       Z-dimension (optional)
    zdim = uint8touint32(b[32:36])

    pixelmap = array.array("f")
    with open(vpxfile, "rb") as f:
        pixelmap.fromfile(f, xdim*ydim*zdim+25)
    if sys.byteorder == "little":
        pixelmap.byteswap()

    pixelmap = pixelmap[25::]
    pixelmap = np.reshape(pixelmap,[zdim,ydim,xdim])

    return pixelmap

def numpy2vpx(pixelmap, VPXfilename):
    datatype = 7
    ndim = len(pixelmap.shape)
    
    zdim, ydim, xdim = pixelmap.shape
    npxls = xdim*ydim*zdim
    
    maxvl = np.amax(pixelmap)
    minvl = np.amin(pixelmap)

    # Create an array of H 16-bit unsigned integers, B 8 bitu
    a = array.array('B', "0"*(114+4*npxls))    
    
    a[0]=ord('V')
    a[1]=ord('P')
    a[2]=ord('X')
    #header[3]     = 0x1   file type (1=Regular pixel file, 2=Pictorial index file)
    a[3] = 1
    #header[4]     = 0x1   version number
    a[4] = 1
    #header[5]     =       data type: byte=2, short=3, float=7, double=8
    a[5] = datatype # 4 = 16 bit
    #header[6]     = 0x8   flag bits
    a[6] = 8
    #header[7]     =       nr of dimensions: 1..7
    a[7] = ndim
    #header[8-15]  =       minimum   (used data type depends on header[5])
    a[8:15] = float32touint8(minvl)
    #header[16-23] =       maximimum (")
    a[16:23] = float32touint8(maxvl)
    #header[24-27] =       X-dimension
    a[24:27] = uint32touint8(xdim)
    #header[28-31] =       Y-dimension (optional)
    a[28:31] = uint32touint8(ydim)
    #header[32-35] =       Z-dimension (optional)
    a[32:35] = uint32touint8(zdim)
    #header[87]    = 0x64 
    a[87] = 100

    #header[96-99] =       freespace table offset
    #                       short : 100 + 2 * #voxels
    #                       float : 100 + 4 * #voxels
    #                       double: 100 + 8 * #voxels
    a[96:99] = uint32touint8(100 + npxls*32/8)

    # Write to file in big endian order
    #for element in pixelmap.flatten():
    flat=pixelmap.flatten()
    buf = struct.pack('>%sf' % len(flat), *flat)
    arr = array.array("B", buf)
    a[100:100+4*npxls] = arr
    
    #for element in xrange(0, xdim*ydim*zdim):
    #    a[100+element*4:104+element*4] = float32touint8(flat[element])
    
    #tail[0]  = 0           ?
    #tail[1]  = 0           ?
    #tail[2]  = 0           ?
    #tail[3]  = 0x1         ?
    #tail[4]  = 0           ?
    #tail[5]  = 0           ?
    #tail[6]  = 0           ?
    #tail[7]  = 0x64        ?
    a[npxls*4+107] = 0x64
    #tail[8]  = 0           ?
    #tail[9]  = 0           ?
    #tail[10] = 0           ?
    #tail[11] =             dim=2  short: 0x8   float: 0x10  double: 0x20
    #                       dim=3  short: 0x10  float: 0x20  double: 0x40
    a[npxls*4+111] = 0x20

    if sys.byteorder == "little":
        a.byteswap()
    with open(VPXfilename, "wb") as f: # wb voor overschrijven, a+b voor appent
        a.tofile(f)



