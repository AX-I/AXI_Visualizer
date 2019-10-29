# Objects for 3D visualizer

from math import sin, cos, pi, log2, ceil
import numpy as np
import time
from Utils import anglesToCoords
from PIL import Image
import random, string
import os, shutil

def rgbToString(rgb):
    return "#{0:02x}{1:02x}{2:02x}".format(*rgb)

class VertObject:    
    def __init__(self, *args, texture=None, texMode="default", gamma=True,
                 **kwargs):
        self.numWedges = 0
        self.estWedges = 0
        self.enabled = True

        self.castShadow = False
        self.receiveShadow = False

        self.texMode = texMode
        self.cgamma = gamma

        self.texMul = 1
        if "texMul" in kwargs:
            self.texMul = kwargs["texMul"]
        
        self.mip = False
        self.reflection = False

        self.useAlpha = False
        
        self.wedgePoints = []
        self.vertNorms = []
        self.wedgePos = []
        self.u = []
        self.v = []
        self.origin = False
        self.static = False

        self.animated = False
        self.bones = None
        
        self.viewer = args[0]
        self.coords = np.array(args[1], dtype="float")
        self.scale = np.array([1.,1,1])
        self.angles = np.array([0.,0,0])
        self.rotMat = np.array([[1., 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])

        self.boneOffset = 0
        if "boneOffset" in kwargs:
            self.boneOffset = kwargs["boneOffset"]
        
        if "mip" in kwargs:
            self.mip = True
        
        if "shadow" in kwargs:
            self.castShadow = "C" in kwargs["shadow"]
            self.receiveShadow = "R" in kwargs["shadow"]
        if "reflect" in kwargs:
            self.reflection = kwargs["reflect"]

        self.overrideNorms = None
        if "overrideNorms" in kwargs:
            self.overrideNorms = np.expand_dims(kwargs["overrideNorms"], 0)
        
        if "alpha" in kwargs:
            af = kwargs["alpha"]
            if af in self.viewer.vaNames:
                pass
            else:
                ti = Image.open(af).rotate(-90)
                if ti.mode == "1":
                    ta = np.array(ti).astype("bool")
                elif ti.mode == "RGBA":
                    ta = np.array(ti)[:,:,3]
                    ta = (ta / 255).astype("bool")
                elif ti.mode == "RGB":
                    ta = np.array(ti.convert("L").convert("1")).astype("bool")
                else:
                    raise ValueError("Unknown alpha mode " + ti.mode)
                self.viewer.texAlphas.append(ta)
                self.viewer.vaNames[af] = len(self.viewer.texAlphas)
            self.useAlpha = self.viewer.vaNames[af]
        
        if texture is None:
            raise ValueError("Texture must be supplied!")
        self.texName = texture

        if "newTexName" in kwargs:
            rr = "".join(random.choices(string.ascii_letters + string.digits, k=4))
            newname = ".".join(texture.split(".")[:-1]) + "_" + \
                      rr + "." + texture.split(".")[-1]
            shutil.copyfile(texture, newname)
            texture = newname

        if texture in self.viewer.vtNames:
            pass
        else:
            ti = Image.open(texture).convert("RGBA").rotate(-90)
            if ti.size[0] != ti.size[1]:
                print("Texture is not square, resizing up.")
                n = max(ti.size)
                ti = ti.resize((n,n))
            if (ti.size[0] & (ti.size[0] - 1)) != 0:
                print("Texture is not a power of 2, resizing up.")
                n = 2**ceil(log2(ti.size[0]))
                ti = ti.resize((n,n))
            ta = np.array(ti).astype("float")
            if self.cgamma:
                ta *= ta # Gamma correction
            else:
                ta *= 256
            ta *= self.texMul
            np.clip(ta, None, 256*256-1, ta)
            self.viewer.vtextures.append(ta.astype("uint16"))
            self.viewer.vtNames[texture] = len(self.viewer.vtextures) - 1
            
            self.viewer.vertPoints.append([])
            self.viewer.vertNorms.append([])
            self.viewer.vertU.append([])
            self.viewer.vertV.append([])
            self.viewer.vertBones.append([])
            
            self.viewer.texMip.append(self.mip)
            self.viewer.texShadow.append(self.receiveShadow)
            
            self.viewer.texUseAlpha.append(self.useAlpha)
            self.viewer.texRefl.append(self.reflection)
            
        self.texNum = self.viewer.vtNames[texture]
        
        if "enabled" in kwargs:
            self.enabled = kwargs["enabled"]
        if "scale" in kwargs:
            self.scale = np.array(kwargs["scale"])
        if "rot" in kwargs:
            rr = kwargs["rot"]
            rotX = np.array([[1, 0, 0],
                                [0, cos(rr[0]), -sin(rr[0])],
                                [0, sin(rr[0]), cos(rr[0])]])
            rotY = np.array([[cos(rr[1]), 0, sin(rr[1])],
                                [0, 1, 0],
                                [-sin(rr[1]), 0, cos(rr[1])]])
            rotZ = np.array([[cos(rr[2]), -sin(rr[2]), 0],
                                [sin(rr[2]), cos(rr[2]), 0],
                                [0, 0, 1]])
            self.rotMat = rotX @ rotZ @ rotY
        if "origin" in kwargs:
            self.origin = kwargs["origin"]

        if "static" in kwargs:
            self.static = True

        self.fitDim = 0
        if "fitDim" in kwargs:
            self.fitDim = kwargs["fitDim"]

    def created(self, early=True):
        self.create()
        #print(self.texName)
        self.cStart = len(self.viewer.vertPoints[self.texNum])
        self.cEnd = self.cStart + self.numWedges
        self.wedgePoints = np.array(self.wedgePoints)
        if self.overrideNorms is None:
            self.vertNorms = np.array(self.vertNorms).reshape((-1, 3))
            self.vertNorms /= np.expand_dims(np.linalg.norm(self.vertNorms, axis=1), 1)
            self.vertNorms = self.vertNorms.reshape((-1, 3, 3))
        else:
            self.vertNorms = np.repeat(self.overrideNorms,
                                          self.wedgePoints.shape[0]*3, 0)
            self.vertNorms = self.vertNorms.reshape((-1, 3, 3))
        self.wedgePos = np.mean(self.wedgePoints, axis=1)
        self.u = np.array(self.u)
        self.v = np.array(self.v)
        if self.texMode == "repeat":
            self.u %= 1
            self.v %= 1
        elif self.texMode == "safe":
            self.u *= 0.999
            self.v *= 0.999
        elif self.texMode == "clamp":
            np.clip(self.u, 0, 1, out=self.u)
            np.clip(self.v, 0, 1, out=self.v)
        if self.fitDim > 0:
            wp = self.wedgePoints.reshape((-1,3))
            ss = np.max(np.abs(np.max(wp, axis=0) - np.min(wp, axis=0)))
            self.scale = self.fitDim / ss
        self.itransform(origin=self.origin, early=early)
        del self.u, self.v
        if self.static:
            del self.wedgePoints, self.vertNorms, self.wedgePos

    def itransform(self, origin=False, early=False):
        tn = self.texNum
        if origin is False:
            newpoints = (self.wedgePoints * self.scale) @ self.rotMat + self.coords            
            newnorms = self.vertNorms @ self.rotMat
        else:
            origin = np.expand_dims(origin, 0)
            newpoints = ((self.wedgePoints-origin) * self.scale) @ self.rotMat + origin + self.coords
            newnorms = self.vertNorms @ self.rotMat
        
        self.viewer.vertPoints[tn].extend(newpoints)
        self.viewer.vertNorms[tn].extend(newnorms)
        self.viewer.vertU[tn].extend(self.u)
        self.viewer.vertV[tn].extend(self.v)
        if self.animated:
            if len(self.viewer.vertBones[tn]) > 0:
                self.viewer.vertBones[tn] = np.concatenate(
                    (self.viewer.vertBones[tn],
                     np.array(self.bones) + self.boneOffset), axis=0)
            else:
                self.viewer.vertBones[tn] = np.array(self.bones) + self.boneOffset

        if not early:
            self.viewer.vertPoints[tn] = np.array(self.viewer.vertPoints[tn])
            self.viewer.vertNorms[tn] = np.array(self.viewer.vertNorms[tn])
            self.viewer.vertU[tn] = np.array(self.viewer.vertU[tn])
            self.viewer.vertV[tn] = np.array(self.viewer.vertV[tn])
        
        self.oldRM = np.array(self.rotMat.T)

    def transform(self, origin=False):
        if origin is False:
            origin = np.array((0.,0.,0.))
        origin += self.coords
        cStart, cEnd = self.cStart*3, self.cEnd*3
        self.viewer.draw.rotate(self.oldRM, self.rotMat, origin,
                                cStart, cEnd, tn)

        self.oldRM = np.array(self.rotMat.T)
    
    def appendWedge(self, coords, norms, uv):
        self.wedgePoints.append(coords)
        self.vertNorms.append(norms)
        self.u.append(uv[:,0])
        self.v.append(uv[:,1])
        self.numWedges += 1

    def appendWedgeSafe(self, coords, norms, uv, r=0):
        if (((np.max(coords, axis=0) -
              np.min(coords, axis=0)) > self.maxWedgeDims).any() and (r < 2)):
            #print(r, end="")
            #if r > 5:
            #    print("test", self.texNum)
            nc, nn, nuv = self.splitFace(coords, norms, uv)
            for f in range(4):
                self.appendWedgeSafe(nc[f], nn[f], nuv[f], r+1)
        else:
            self.wedgePoints.append(coords)
            self.vertNorms.append(norms)
            self.u.append(uv[:,0])
            self.v.append(uv[:,1])
            self.numWedges += 1

    def splitFace(self, c, n, uv):
        nc = (c + np.roll(c, -1, 0)) / 2
        nn = (n + np.roll(n, -1, 0)) / 2
        nn = nn / np.expand_dims(np.linalg.norm(nn, axis=1), axis=1)
        nuv = (uv + np.roll(uv, -1, 0)) / 2
        fc = np.array([(c[0], nc[0], nc[2]), (c[1], nc[1], nc[0]),
                       (c[2], nc[2], nc[1]), nc])
        fn = np.array([(n[0], nn[0], nn[2]), (n[1], nn[1], nn[0]),
                       (n[2], nn[2], nn[1]), nn])
        fuv = np.array([(uv[0], nuv[0], nuv[2]), (uv[1], nuv[1], nuv[0]),
                        (uv[2], nuv[2], nuv[1]), nuv])
        
        return fc, fn, fuv

    def rotate(self, rr):
        rotX = np.array([[1, 0, 0],
                            [0, cos(rr[0]), -sin(rr[0])],
                            [0, sin(rr[0]), cos(rr[0])]])
        rotY = np.array([[cos(rr[1]), 0, sin(rr[1])],
                            [0, 1, 0],
                            [-sin(rr[1]), 0, cos(rr[1])]])
        rotZ = np.array([[cos(rr[2]), -sin(rr[2]), 0],
                            [sin(rr[2]), cos(rr[2]), 0],
                            [0, 0, 1]])
        self.rotMat = rotX @ rotZ @ rotY


class VertWater(VertObject):
    def __init__(self, *args, size=60, pScale=4, wDir=(0.8,0.6),
                 wLen=[10], wAmp=[1], wSpd=[1], numW=1, **ex):
        super().__init__(*args, **ex)

        self.wDir = np.array(wDir)
        self.pS = pScale
        self.wLen = np.array(wLen)
        self.wAmp = np.array(wAmp)
        self.wSpd = np.array(wSpd)
        self.numW = numW
        self.size = size
        self.estWedges = size*size*2
        self.hasSetup = False

    def create(self):
        n = np.array([[0, 1., 0]] * 3)
        for i in range(self.size):
            for j in range(self.size):
                wc = np.array([(i,0.,j), (i+1,0,j), (i,0,j+1)])
                tx = np.array([(0., 0), (1, 0), (0, 1)])
                self.appendWedge(wc, n, tx)

                wc = np.array([(i+1,0.,j+1), (i,0,j+1), (i+1,0,j)])
                tx = np.array([(1., 1), (0, 1), (1, 0)])
                self.appendWedge(wc, n, tx)
        self.stTime = time.time()
        
    def update(self):
        if not self.hasSetup:
            self.viewer.draw.setupWave(self.coords, self.wDir,
                                       self.wLen, self.wAmp, self.wSpd,
                                       self.numW)
            self.hasSetup = True
        self.viewer.draw.updateWave(self.pS, self.stTime, self.texNum)

class VertSphere(VertObject):
    def __init__(self, *args, n=16, size=1, **ex):
        super().__init__(*args, **ex)
        
        self.dims = {"n":n, "size":size}
        self.n = n
        self.estWedges = n * (n-1) * 2

    def create(self):
        self.numWedges = 0
        pos = np.array(self.coords)
        n = self.n
        self.pts = []
        
        for i in range(1, n):
            a = (i/n - 0.5)*pi
            p = []
            for j in range(n):
                b = j*2*pi/n
                tc = anglesToCoords([b, a])
                p.append(tc)
            self.pts.append(p)

        self.np = np.array([0, 1, 0])
        self.sp = np.array([0, -1, 0])

        for k in range(n):
            uv = np.array([(1/n, k/n), (0, k/n), (1/n, (k+1)/n)])
            wc = np.array([self.pts[0][k-1], self.sp, self.pts[0][k]])
            self.appendWedge(wc, -wc, uv)
            
        for i in range(n - 3, -1, -1):
            for j in range(n):
                uv = np.array([((i+1)/n, j/n), ((i+1)/n, (j+1)/n), ((i+2)/n, (j+1)/n)])
                wc = np.array([self.pts[i][j-1], self.pts[i][j], self.pts[i+1][j]])

                self.appendWedge(wc, -wc, uv)
                uv = np.array([((i+1)/n, j/n), ((i+2)/n, (j+1)/n), ((i+2)/n, j/n)])
                wc = np.array([self.pts[i][j-1], self.pts[i+1][j], self.pts[i+1][j-1]])
                self.appendWedge(wc, -wc, uv)

        for k in range(n):
            uv = np.array([((n-1)/n, k/n), ((n-1)/n, (k+1)/n), (1, k/n)])
            wc = np.array([self.pts[-1][k-1], self.pts[-1][k], self.np])
            self.appendWedge(wc, -wc, uv)

modelList = {}
class VertModel(VertObject):
    def __init__(self, *args, filename=None, size=1, mtlNum=0,
                 childMtl=0,
                 animated=None, **ex):
        """Loads .obj files"""

        self.root = True
        if childMtl == 1:
            self.root = False

        if "/" in filename:
            self.path = "/".join(filename.split("/")[:-1]) + "/"
        else:
            self.path = ""
        self.mtlName = None
        self.nextMtl = None
        c = False
        if "texture" in ex:
            self.mtlTex = ex["texture"]
            c = True
            del ex["texture"]

        self.late = False
        if "late" in ex:
            self.late = True

        if not c:
            with open(filename) as f:
                f.readline()
                if animated is None:
                    if "# Includes vertex bone assignment" in f.readline():
                        animated = True
                    else:
                        animated = False
            with open(filename) as f:
                for line in f:
                    if (line == "\n") or (line[0] == "#"):
                        continue
                    else:
                        t = line.split()
                        if t[0] == "mtllib":
                            self.mtl = self.path + t[1]
                            break
            cmtl = -1
            alpha = None
            with open(self.mtl) as f:
                for line in f:
                    if (line == "\n") or (line[0] == "#"):
                        continue
                    else:
                        t = line.split()
                        if len(t) == 0: continue
                        if t[0] == "newmtl":
                            cmtl += 1
                            if cmtl == mtlNum:
                                self.mtlName = t[1]
                        elif t[0] == "map_Kd":
                            if cmtl == mtlNum:
                                self.mtlTex = self.path + t[1]
                                self.mtlTexFname = t[1]
                                c = True
                        elif t[0] == "map_d":
                            if cmtl == mtlNum:
                                alpha = self.path + t[1]
            if cmtl > mtlNum:
                exn = dict(ex)
                if not self.late:
                    self.nextMtl = VertModel(*args, filename=filename, size=size,
                                             animated=animated,
                                             mtlNum=mtlNum+1, childMtl=1, **exn)
                    args[0].vertObjects.append(self.nextMtl)
                else:
                    del exn["late"]
                    self.nextMtl = VertModel(*args, filename=filename, size=size,
                                             animated=animated,
                                             mtlNum=mtlNum+1, childMtl=1,
                                             late=1, **exn)
                    args[0].vertObjects.append(self.nextMtl)
        if not c:
            raise ValueError("Texture must be supplied!")

        if alpha is not None:
            ex["alpha"] = alpha
        super().__init__(*args, texture=self.mtlTex, **ex)
        
        self.animated = animated
        
        self.numWedges = 0
        self.filename = filename
        self.size = size
        self.estWedges = 0
        with open(filename) as f:
            for line in f:
                if line[0] == "f":
                    self.estWedges += len(line.split()) - 3
    
    def create(self):        
        filename = self.filename

        global modelList
        if filename in modelList:
            if self.mtlTex in modelList[filename]:
                tmod = modelList[filename][self.mtlTex]
                self.wedgePoints = tmod["wp"]
                self.vertNorms = tmod["vn"]
                self.u = tmod["u"]
                self.v = tmod["v"]
                self.numWedges = tmod["nw"]
                return
        
        size = self.size

        self.points = []
        self.vts = []
        self.vns = []
        if self.animated:
            self.bones = []
        activeMat = False
        with open(filename) as f:
            line = f.readline()
            while not (line == ""):
                if line[0] == "#":
                    line = f.readline()
                elif (line == "\n"):
                    line = f.readline()
                else:
                    t = line.split()
                    if t[0] == "v":
                        a = [float(s) for s in t[1:4]]
                        a[2] = -a[2]
                        self.points.append(a)
                    elif t[0] == "f":
                        if activeMat:
                            c = [self.points[int(s.split("/")[0]) - 1] for s in t[1:]]
                            tx = [self.vts[int(s.split("/")[1]) - 1] for s in t[1:]]
                            n = [self.vns[int(s.split("/")[2]) - 1] for s in t[1:]]
                            self.appendWedge(np.array(c), -np.array(n), np.array(tx))
                            #self.appendWedgeSafe(np.array(c), -np.array(n), np.array(tx))
                            if self.animated:
                                self.bones.append([int(s.split("/")[3]) for s in t[1:]])
                    elif t[0] == "vt":
                        # uv is actually vu
                        self.vts.append((float(t[2]), float(t[1])))
                    elif t[0] == "vn":
                        n = [float(s) for s in t[1:4]]
                        n[2] = -n[2]
                        self.vns.append(n)
                    elif t[0] == "usemtl":
                        if t[1] == self.mtlName:
                            activeMat = True
                        else:
                            activeMat = False
                    line = f.readline()

        if filename not in modelList:
            modelList[filename] = {}
        elif self.mtlTex not in modelList[filename]:
            tm = {"wp":self.wedgePoints, "vn":self.vertNorms,
                  "u":self.u, "v":self.v, "nw": self.numWedges}
            modelList[filename][self.mtlTex] = tm
        
        del self.vns, self.vts

    def rotateAll(self, rr):
        if self.nextMtl is not None:
            self.nextMtl.rotateAll(rr)
        self.rotate(rr)
    def translateAll(self, cc):
        if self.nextMtl is not None:
            self.nextMtl.translateAll(cc)
        self.coords = cc
    
    def transformAll(self, origin=False, early=False):
        if self.nextMtl is not None:
            self.nextMtl.transformAll(origin, early)
        self.transform(origin, early)
                    
class VertTerrain(VertObject):
    def __init__(self, *args, size=(20,20), heights=None, uvspread=2, **ex):
        """heights -> array dims (size+1, size+1) OR filename"""
        super().__init__(*args, **ex)
        
        self.size = np.array(size)
        self.heights = heights

        if type(heights) is str: #Filename
            himg = Image.open(heights)
            if himg.mode == "F":
                self.heights = np.array(himg)
            else:
                gr = himg.convert("L")
                hmap = np.array(gr)
                self.heights = hmap / 255
            self.size = np.array(himg.size) - 1
        elif heights is None:
            self.heights = np.zeros(self.size + 1)
        else:
            self.heights = heights
            self.size = np.array(heights.shape) - 1
            
        self.estWedges = self.size[0] * self.size[1] * 2

        self.uvspread = uvspread
        self.texMode = "repeat"

    def create(self):
        self.heights = np.array(self.heights)
        
        self.pts = []
        for i in range(self.size[0] + 1):
            ri = np.repeat(i, self.size[1]+1)
            rj = np.arange(self.size[1]+1)
            row = np.stack((ri, self.heights[i], rj)).T
            self.pts.append(row)
        
        self.pts = np.array(self.pts)
        
        self.norms = np.empty((self.size[0] + 1, self.size[1] + 1, 3),
                                 dtype="float")
        self.norms[0,:] = [0, -1, 0]
        self.norms[-1,:] = [0, -1, 0]
        self.norms[:,0] = [0, -1, 0]
        self.norms[:,-1] = [0, -1, 0]
        oo = np.ones((self.size[0]-1,))
        zz = np.zeros((self.size[0]-1,))
        p = self.heights[1][1:-1]
        p1 = self.heights[0][1:-1]
        p3 = self.heights[2][1:-1]
        for i in range(1, self.size[0]):
            p1 = p
            p = p3
            p2 = self.heights[i][:-2]
            p3 = self.heights[i+1][1:-1]
            p4 = self.heights[i][2:]
            v1 = np.stack((oo, p-p1, zz))
            v2 = np.stack((zz, p-p2, oo))
            v3 = np.stack((-oo, p-p3, zz))
            v4 = np.stack((zz, p-p4, -oo))
            n1 = np.cross(v1, v2, axis=0)
            n2 = np.cross(v2, v3, axis=0)
            n3 = np.cross(v3, v4, axis=0)
            n4 = np.cross(v4, v1, axis=0)
            nn = np.stack((n1, n2, n3, n4))
            self.norms[i][1:-1] = np.average(nn, axis=0).T

        sp = self.uvspread
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                wc = np.array([self.pts[i][j], self.pts[i+1][j], self.pts[i][j+1]])
                n = np.array([self.norms[i][j], self.norms[i+1][j], self.norms[i][j+1]])
                tx = np.array([(i, j), (i+0.99, j), (i, j+0.99)])
                self.appendWedge(wc, n, tx/sp)

                wc = np.array([self.pts[i+1][j+1], self.pts[i][j+1], self.pts[i+1][j]])
                n = np.array([self.norms[i+1][j+1], self.norms[i][j+1], self.norms[i+1][j]])
                tx = np.array([(i+0.99, j+0.99), (i, j+0.99), (i+0.99, j)])
                self.appendWedge(wc, n, tx/sp)

    def getHeight(self, x, z):
        """World coords x,z -> world coord y"""
        landCoord = (np.array([x, 0, z]) - self.coords) / self.scale
        #h = self.heights[int(landCoord[0]), int(landCoord[2])+4]
        texr1 = landCoord[0]
        tex1 = int(texr1)
        texr1 -= tex1
        texi1 = 1-texr1
        texr2 = landCoord[2]
        tex2 = int(texr2)
        texr2 -= tex2
        texi2 = 1-texr2
        h0 = self.heights[tex1, tex2]
        h1 = self.heights[min(tex1+1, self.heights.shape[0]-1), tex2]
        h2 = self.heights[tex1, min(tex2+1,self.heights.shape[1]-1)]
        h3 = self.heights[min(tex1+1, self.heights.shape[0]-1),
                          min(tex2+1, self.heights.shape[1]-1)]
        h = h0*texi1*texi2 + h1*texr1*texi2 + h2*texi1*texr2 + h3*texr1*texr2
        try:
            if self.scale.shape[0] == 3:
                s = self.scale[1]
        except:
            s = self.scale
        return h * s + self.coords[1]
    
class VertPlane(VertObject):
    name="Plane"
    def __init__(self, *args, n=8, h1=[4, 0, 0], h2=[0, 8, 0],
                 norm=False, **ex):
        super().__init__(*args, **ex)
        self.estWedges = n**2 * 2
        self.h1 = np.array(h1, dtype="float64")
        self.h2 = np.array(h2, dtype="float64")
        if norm is False:
            normal = np.cross(self.h1, self.h2)
            normal /= np.linalg.norm(normal)
        else:
            normal = np.array(norm, dtype="float64")
        self.normal = -normal
        self.n = n

    def create(self):
        n = self.n
        m1 = self.h1/n
        m2 = self.h2/n
        norm = np.repeat([self.normal], 3, 0)
        for i in range(n):
            for j in range(n):
                c = m1*i + m2*j
                coords = np.array([c, c + m2, c + m1])
                uv = np.array([(i/n, j/n), (i/n, (j+1)/n), ((i+1)/n, j/n)])
                coords2 = np.array([c + m1 + m2, c + m1, c + m2])
                uv2 = np.array([((i+1)/n, (j+1)/n),
                                   ((i+1)/n, j/n), (i/n, (j+1)/n)])
                self.appendWedge(coords, norm, uv)
                self.appendWedge(coords2, norm, uv2)
