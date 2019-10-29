# Textured Objects for 3D visualizer

from math import sin, cos, pi
import numpy
from PIL import Image

class TexSkyBox:
    def __init__(self, v, c, N, tex, rot=(0,0,0)):
        self.viewer = v
        self.pts = []
        self.texIndex = 0

        ti = Image.open(tex).convert("RGB")
        m = ti.size[1]
        if m*6 == ti.size[0]:
            self.bottom = 1
        elif m*5 == ti.size[0]:
            self.bottom = 0
        else:
            raise ValueError("Image is not w:h = 6:1 or 5:1!")

        self.numWedges = 6 * m**2 * 2
        self.N = N + 1
        
        ti = Image.open(tex).convert("RGBA")
        ta = numpy.array(ti).astype("float")
        ta *= ta
        ta = ta.transpose((1, 0, 2))
        self.viewer.skyTex = ta.astype("uint16")

        rr = rot
        rotX = numpy.array([[1, 0, 0],
                            [0, cos(rr[0]), -sin(rr[0])],
                            [0, sin(rr[0]), cos(rr[0])]])
        rotY = numpy.array([[cos(rr[1]), 0, sin(rr[1])],
                            [0, 1, 0],
                            [-sin(rr[1]), 0, cos(rr[1])]])
        rotZ = numpy.array([[cos(rr[2]), -sin(rr[2]), 0],
                            [sin(rr[2]), cos(rr[2]), 0],
                            [0, 0, 1]])
        self.rotMat = rotX @ rotZ @ rotY
    
    def created(self):
        N = self.N
        bb = 5 + self.bottom

        for i in range(N):
            a = i/(N-1) - 0.5
            for j in range(N):
                b = j/(N-1) - 0.5
                c = (a, b, 0.5)
                self.pts.append(c)
        self.pts = numpy.array(self.pts)
        
        t1 = self.pts[:,[2,1,0]]
        t2 = self.pts[:,[1,2,0]]
        r1 = numpy.array((1, -1, -1))
        r2 = numpy.array((1, -1, 1))
        r3 = numpy.array((1, 1, -1))
        r4 = numpy.array((-1, -1, 1))
        self.pts = numpy.array([r2*self.pts, r1*t1, -1*self.pts, r4*t1,
                                r3*t2, -1*t2]).reshape((bb, N, N, 3))

        self.pts = self.pts.reshape((-1,3))
        self.pts = self.pts @ self.rotMat
        self.pts = self.pts.reshape((bb, N, N, 3))

        M = N - 1
        for x in range(bb):
            for j in range(N-1):
                for i in range(N-1):
                    wc = numpy.array([self.pts[x][i][j],
                                      self.pts[x][i+1][j],
                                      self.pts[x][i][j+1]])
                    cc = numpy.array([((x + i/M)    , j/M),
                                      ((x + (i+1)/M), j/M),
                                      ((x + i/M)    , (j+1)/M)])
                    self.appendWedge(wc, cc)
                
                    w2 = numpy.array([self.pts[x][i+1][j+1],
                                      self.pts[x][i][j+1],
                                      self.pts[x][i+1][j]])
                    c2 = numpy.array([((x + (i+1)/M), (j+1)/M),
                                      ((x + i/M)    , (j+1)/M),
                                      ((x + (i+1)/M), j/M)])
                    self.appendWedge(w2, c2)

    def appendWedge(self, coords, uv):
        self.viewer.skyPoints.append(coords)
        avg = numpy.average(coords, axis=0)
        self.viewer.skyAvgPos.append(avg / numpy.linalg.norm(avg))
        self.viewer.skyU.append(uv[:,0])
        self.viewer.skyV.append(uv[:,1])

