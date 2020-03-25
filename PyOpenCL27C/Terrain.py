# Terrain generator
# Hash octaves

import numpy as np
import numpy.random as nr
from PIL import Image
import hashlib
from math import floor, ceil, sin, pi

class RandTerrain:
    def __init__(self, seed, period=32, octaves=6, scale=1):
        self.s = str(seed)
        self.p = period
        self.o = octaves
        self.sc = scale

    def randCoord(self, x, y, octave=0):
        """Octave 0 to 7, returns 0 to 32768"""
        b = bytes(self.s + str(x) + "," + str(y), "ascii")
        return int(hashlib.md5(b).hexdigest()[4*octave:4*(octave+1)], 16) // 2

    def cubic(self, t, a1, a2, a3, a4):
        a = np.array([[a1, a2, a3, a4]]).T
        x = np.array([[0,2,0,0],[-1,0,1,0],[2,-5,4,-1],[-1,3,-3,1]])
        m = np.array([[1,t,t*t,t*t*t]])
        return 0.5 * m @ x @ a

    def getHeight(self, x, y):
        """Bicubic, 0 to 65535"""
        h = 0
        p = self.p
        rc = self.randCoord
        for i in range(self.o):
            tx = x/p - int(x/p)
            ty = y/p - int(y/p)
            b = []
            for j in range(-1,3):
                b.append(self.cubic(tx, rc(floor(x/p)-1, floor(y/p)+j),
                                    rc(floor(x/p), floor(y/p)+j),
                                    rc(floor(x/p)+1, floor(y/p)+j),
                                    rc(floor(x/p)+2, floor(y/p)+j)))
            r = self.cubic(ty, *b)
            h += self.sc / (2**i) * r
            p /= 2
        return h

    def getHeight0(self, x, y):
        """Bilinear, 0 to 65535"""
        h = 0
        p = self.p
        for i in range(self.o):
            tx = x/p - int(x/p)
            ty = y/p - int(y/p)
            r = (1-tx)*(1-ty)*self.randCoord(floor(x/p), floor(y/p))
            r += (1-tx)*ty*self.randCoord(floor(x/p), floor(y/p)+1)
            r += tx*(1-ty)*self.randCoord(floor(x/p)+1, floor(y/p))
            r += tx*ty*self.randCoord(floor(x/p)+1, floor(y/p)+1)
            h += self.sc / (2**i) * r
            p /= 2
        return h

    def getArea(self, x0, x1, y0, y1, res, offset=0):
        a = np.zeros((ceil((y1-y0)/res), ceil((x1-x0)/res)))
        i1 = 0; j1 = 0
        for i in np.arange(y0, y1, res):
            for j in np.arange(x0, x1, res):
                a[i1,j1] = self.getHeight(i, j)
                j1 += 1
            i1 += 1
            j1 = 0
        return a - offset

def saveTerrain(t, f):
    a = Image.fromarray(t)
    a.save(f)
