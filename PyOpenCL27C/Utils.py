# Random utilities for 3D

from numpy import sqrt, ones_like, array
from math import sin, cos, pi, asin, acos, log2
import numexpr as ne
import numpy as np

def viewVec(a, b):
    v = np.array([sin(a) * cos(b),
                  -sin(b),
                  cos(a) * cos(b)])
    return v

def viewMat(a, b):
    a2 = a + pi/2
    b2 = b - pi/2
    v = np.array([[sin(a)*cos(b),   -sin(b),  cos(a)*cos(b)],
                  [-sin(a2),         0,        -cos(a2)],
                  [-sin(a)*cos(b2),  sin(b2),  -cos(a)*cos(b2)]])
    return v

def viewMat2(a, b):
    a2 = a + pi/2
    b2 = b - pi/2
    v = np.array([[sin(a2),         0,        cos(a2)],
                  [sin(a)*cos(b2),  -sin(b2),  cos(a)*cos(b2)]])
    return v

def processWedgesDepth(vM, vc, cullAngle, wedgepoints, scale, size):
    vWs = wedgepoints.reshape((-1, 3)) - vc
    dxy = vWs @ vM.T
    dxy = dxy.reshape((-1, 3, 3))

    d = 1 / dxy[:,:,0]
    x = dxy[:,:,1]
    y = dxy[:,:,2]
    cA = cullAngle
    visibleS = ne.evaluate("((d > 0) & (abs(x*d) < cA) & (abs(y*d) < cA))")
    visibleS = visibleS.all(axis=1)
    #n = ((wedgenorms @ vM[0]) < 0).any(axis=1)
    #visibleS = visibleS & n

    d = d[visibleS].reshape((-1,))
    x = x[visibleS].reshape((-1,))
    y = y[visibleS].reshape((-1,))

    ne.evaluate("x * scale + size", out=x)
    ne.evaluate("y * -scale + size", out=y)

    xy2d = np.stack((x, y), axis=1)
    
    return (xy2d, d)

def processWedgesDepthUV(vM, vc, cullAngle, wedgepoints,
                         ui, vi, scale, size):
    vWs = wedgepoints.reshape((-1, 3)) - vc
    dxy = vWs @ vM.T
    dxy = dxy.reshape((-1, 3, 3))

    d = 1 / dxy[:,:,0]
    x = dxy[:,:,1]
    y = dxy[:,:,2]
    cA = cullAngle
    visibleS = ne.evaluate("((d > 0) & (abs(x*d) < cA) & (abs(y*d) < cA))")
    visibleS = visibleS.all(axis=1)

    d = d[visibleS].reshape((-1,))
    x = x[visibleS].reshape((-1,))
    y = y[visibleS].reshape((-1,))

    ne.evaluate("x * scale + size", out=x)
    ne.evaluate("y * -scale + size", out=y)

    xy2d = np.stack((x, y), axis=1)
    u = ui[visibleS]
    v = vi[visibleS]
    return (xy2d, d, u, v)

def createMip(a):
    b = a[:-1:2] + a[1::2]
    c = b[:,:-1:2] + b[:,1::2]
    return c >> 2

def createMips(ar):
    """d -> debug / display images"""
    a = np.array(ar).astype("int")
    m = [a]
    for i in range(int(log2(ar.shape[0]))):
        a = createMip(a)
        m.append(a)
    m.reverse()
    dim = a.shape[-1]
    return np.concatenate([x.reshape(-1,dim) for x in m], axis=0)
    
def eucDist(a, b):
    c = (a - b) ** 2
    return sqrt(c @ ones_like(c))

def nenorm(a, axis):
    return sqrt(ne.evaluate("sum(a*a," + str(axis) + ")"))
    
def divsqrt32(b, a):
    """Approximates b / sqrt(a)"""
    c = np.int32(0x5f3759df)
    #th = np.float32(1.5)
    #hf = np.float32(0.5)
    i = a.view(np.int32)
    i = ne.evaluate("c - (i >> 1)")
    y = i.view(np.float32)
    ne.evaluate("b * y", out=b)
    #ne.evaluate("b * y * (th - (a*hf * y*y))", out=b)

def divsqrt64(b, a):
    """Approximates b / sqrt(a)"""
    c = np.int64(0x5FE6EB50C7B537A9)
    i = a.view(np.int64)
    i = ne.evaluate("c - (i >> 1)")
    y = i.view(np.float64)
    ne.evaluate("b * y", out=b)
    #ne.evaluate("b * y * (th - (a*hf * y*y))", out=b)

def anglesToCoords(ab):
    x = cos(ab[0]) * cos(ab[1])
    y = sin(ab[1])
    z = sin(ab[0]) * cos(ab[1])
    return array((x, y, z))

def rgbnToString(rgbn):
    rgb1 = map(lambda xn: [max(0, x) for x in xn], rgbn)
    return ["#{0:02x}{1:02x}{2:02x}".format(*rgb) for rgb in rgb1]

hx = "0123456789abcdef"
def info_rgb(rgb):
    if type(rgb[0]) is int:
        return rgb
    if rgb[0] != "#":
        raise ValueError("Unknown color, use #xxx or #xxxxxx")
    if len(rgb) == 4:
        return (hx.index(rgb[1])*17, hx.index(rgb[2])*17, hx.index(rgb[3])*17)
    if len(rgb) == 7:
        return (hx.index(rgb[1])*16 + hx.index(rgb[2]),
                hx.index(rgb[3])*16 + hx.index(rgb[4]),
                hx.index(rgb[5])*16 + hx.index(rgb[6]))
    raise ValueError("Unknown color " + rgb + ", use #xxx or #xxxxxx")
