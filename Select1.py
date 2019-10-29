# AXI Visualizer: Animator
# Skeletal animation
# Rig construction and
# Bone vertex painting

# New: Bone/Joint renaming

from tkinter import *
from tkinter.messagebox import *

from math import sin, cos, sqrt, pi
import numpy
import numpy as np
import time
import json
import traceback

from Compute import *
import multiprocessing as mp
import os

from Rig import Rig, Bone

class ThreeDApp(ThreeDBackend):
    def __init__(self, width=960, height=640, fovx=80, fopen=None):
        super().__init__(width, height, fovx=fovx)

        self.fopen = fopen

        self.changeTitle("AXI Visualizer: Animator")

        self.α = -1.5
        self.β = 0.2
        
        self.pos = numpy.array([6.4, 3.1, 1.4])
        self.camSpeed = 0.04

        self.mouseSensitivity = 20

        self.ambLight = 0.15
        self.maxFPS = 33

        self.sphereRot = [0, 0, 0]

        self.selRad = 0.4
        self.selBone = 0
        self.selecting = False

        self.drawAxes = True
        self.moveBone = False

        self.bi = 0
        self.ctexn = []

        self.guiType = "Select"

    def moveKey(self, key):
        if key == "u":   self.svert = 1
        elif key == "d": self.svert = -1
        elif key == "r": self.shorz = 1
        elif key == "l": self.shorz = -1
        elif key == "ZV": self.svert = 0
        elif key == "ZH": self.shorz = 0
        elif key[:4] == "save": self.exportChar(key[4:])
        elif key[:4] == "open": self.importChar(key[4:])
        elif key[:5] == "scale": self.scaleChar(key[5:])
        elif key[:3] == "rot": self.rotChar(key[3:])
        elif key[:6] == "switch": self.switchBone(key[6:])
        elif key[:7] == "newBone": self.newBone()
        elif key[:5] == "bindj": self.bindJoint()

    def customizeFrontend(self):
        self.bindKey("1", "rp"); self.makeHandler("rp", self.rp)
        self.bindKey("2", "rn"); self.makeHandler("rn", self.rn)
        
        self.bindKey("r", "rR"); self.makeHandler("rR", self.rotateLight)
        
        self.bindKey("i", "ix"); self.makeHandler("ix", self.ix)
        self.bindKey("I", "ix2"); self.makeHandler("ix2", self.ix2)
        self.bindKey("k", "iy"); self.makeHandler("iy", self.iy)
        self.bindKey("l", "iz"); self.makeHandler("iz", self.iz)
        self.bindKey("L", "iz2"); self.makeHandler("iz2", self.iz2)

    def ix(self): self.α = 0; self.β = 0
    def ix2(self): self.α = pi; self.β = 0
    def iy(self): self.α = 0; self.β = pi/2
    def iz(self): self.α = pi/2; self.β = 0
    def iz2(self): self.α = -pi/2; self.β = 0
    
    def chooseBone(self):
        self.bi += 1
        if self.bi >= len(self.allBones):
            self.bi = 0
        elif self.bi < 0:
            self.bi += len(self.allBones)
        self.cb = self.allBones[self.bi]
        self.axPoints = self.cb.getPoints(self.baseAxPoints)
        self.P.put(("bnum", (self.selBone, self.bi, len(self.allBones))))
        self.getText()
        self.P.put(("rnum", self.rigText, self.bi))
    def backBone(self):
        self.bi -= 2
        self.chooseBone()
    def switchBone(self, n):
        self.bi = int(n) - 1
        self.chooseBone()
        
    def newBone(self):
        try: a = self.allBones
        except AttributeError: return
        nb = Bone((0.,0.,0.), bn=len(self.allBones))
        self.cb.addChild(nb)
        nb.isNew = True
        self.allBones.append(nb)
        
        self.P.put(("bnum", (self.selBone, self.bi, len(self.allBones))))
        self.getText()
        self.P.put(("rnum", self.rigText, self.bi))
        
    def rp(self):
        self.selRad *= 1.2
    def rn(self):
        self.selRad /= 1.2
    
    def bp(self):
        self.selBone += 1
        for i in self.ctexn:
            self.draw.highlight(self.selBone, np.array([0,0,0]),
                                0, 0.4, 2, 3, 0, i)
        self.P.put(("bnum", (self.selBone, self.bi, len(self.allBones))))
    def bn(self):
        self.selBone -= 1
        for i in self.ctexn:
            self.draw.highlight(self.selBone, np.array([0,0,0]),
                                0, 0.4, 2, 3, 0, i)
        self.P.put(("bnum", (self.selBone, self.bi, len(self.allBones))))

    def bindJoint(self):
        for i in self.ctexn:
            self.draw.switch(self.selBone, self.bi, i)

        self.selBone = self.bi - 1
        self.bp()
    
    def select(self):
        if self.moveBone:
            self.moveBone = False
        else:
            self.selecting = not self.selecting

    def select2(self):
        if self.selecting & (not self.moveBone):
            self.moveBone = True
        else:
            self.selecting = not self.selecting
            self.moveBone = not self.moveBone

    def bSelect(self, commit, xy):
        cz = (self.cb.getPoints(np.array([(0,0,0)])) - self.pos) @ self.vv
        xy -= np.array((self.W2, self.H2))
        p = self.pos + cz*self.viewVec()
        p += cz* -xy[0]/self.scale * self.vVhorz()
        p += cz* xy[1]/self.scale * self.vVvert()

        aa = self.cb
        while aa.parent is not None:
            p -= aa.parent.origin
            aa = aa.parent

        self.cb.origin = p
        self.cb.offset = np.array((*p,1))
        self.cb.updateTM()
        self.axPoints = self.cb.getPoints(self.baseAxPoints)

        self.drawLines = True
        self.getLines()
        
    def cSelect(self, commit, xy):
        if self.moveBone:
            self.bSelect(commit, xy)
            return
        
        cz = self.zb[xy[1], xy[0]]
        xy -= np.array((self.W2, self.H2))
        p = self.pos + cz*self.viewVec()
        p += cz* -xy[0]/self.scale * self.vVhorz()
        p += cz* xy[1]/self.scale * self.vVvert()
        
        for i in self.ctexn:
            self.draw.highlight(self.selBone, p, self.selRad,
                                0.4, 2, 3, commit, i)

    def getLines(self, r=None):
        if r is None:
            self.linePoints = []
            r = self.rig.b0
        for x in r.children:
            self.linePoints.append((r.getPoints(np.zeros(3)), x.getPoints(np.zeros(3))))
            self.getLines(x)
        if r is self.rig.b0:
            self.linePoints = np.array(self.linePoints).reshape((-1,3))

    def getText0(self, r=None, n=0):
        if r is None:
            self.rigText = "0\n"
            r = self.rig.b0
        for x in r.children:
            #self.rigText += n*"| " + "|-" + str(x.boneNum) + "\n"
            self.rigText += n*"  " + str(x.boneNum) + "\n"
            self.getText(x, n+1)

    def getText(self, r=None, n=0):
        if r is None:
            self.rigText = [("0 (Root)", 0)]
            r = self.rig.b0
        for x in r.children:
            self.rigText.append((4*n*" " + str(x.boneNum), x.boneNum))
            self.getText(x, n+1)

    def importChar(self, fn):
        if fn == "": return
        try:
            self.insertObject(VertModel, [0,0,0],
                              filename=fn, texMode="repeat",
                              scale=1, shadow="C")
        except Exception as e:
            root = Tk()
            root.withdraw()
            showerror("Error", "Couldn't load model:\n"+traceback.format_exc())
            root.destroy()
            return
        self.testv = self.vertObjects[-1]

        ctexn = []
        c = self.testv
        ctexn.append(c.texNum)
        while c.nextMtl is not None:
            c = c.nextMtl
            ctexn.append(c.texNum)
        self.ctexn = ctexn

        try:
            self.createInserted(reversed(ctexn))
        except Exception as e:
            root = Tk()
            root.withdraw()
            showerror("Error", "Couldn't load model:\n"+traceback.format_exc())
            root.destroy()
            return

        if self.testv.animated:
            try:
                self.initRig(".".join(fn.split(".")[:-1]) + ".rig")
            except FileNotFoundError:
                self.rig = Rig({"boneNum":0, "origin":[0,0,0], "children":[]})
                self.allBones = self.rig.allBones
                self.b0 = self.rig.b0
        else:
            self.rig = Rig({"boneNum":0, "origin":[0,0,0], "children":[]})
            self.allBones = self.rig.allBones
            self.b0 = self.rig.b0
            
            self.vertBones = []
            for i in self.vertLight:
                self.vertBones.append(np.zeros((i.shape[0], 3), dtype="int"))
        
        for i in range(len(self.vertBones)):
            if len(self.vertBones[i]) > 0:
                self.draw.addBoneWeights(i, self.vertBones[i])

        self.cb = self.allBones[self.bi]

        self.pendingShader = True
        self.rotateLight()

        self.bindKey("e", "eE"); self.makeHandler("eE", self.select)
        self.bindKey("f", "fF"); self.makeHandler("fF", self.chooseBone)
        self.bindKey("F", "fg"); self.makeHandler("fg", self.backBone)
        self.bindKey("g", "gG"); self.makeHandler("gG", self.newBone)
        self.bindKey("t", "tT"); self.makeHandler("tT", self.select2)
        self.bindKey("4", "tl"); self.makeHandler("tl", self.toggleLines)
        self.bindKey("z", "bp"); self.makeHandler("bp", self.bp)
        self.bindKey("x", "bn"); self.makeHandler("bn", self.bn)

        self.P.put(("bnum", (self.selBone, self.bi, len(self.allBones))))

        self.getText()
        self.P.put(("rnum", self.rigText, self.bi))

        self.drawLines = True
        self.getLines()

    def toggleLines(self):
        self.drawLines = not self.drawLines

    def scaleChar(self, s):
        try: s = float(s)
        except: return
        for i in self.ctexn:
            self.draw.scale(np.array([0,0,0]), s, i)

    def rotChar(self, ax):
        if ax == "X": rm = rotMat((pi/2,0,0))
        elif ax == "Y": rm = rotMat((0,pi/2,0))
        elif ax == "Z": rm = rotMat((0,0,pi/2))
        for i in self.ctexn:
            self.draw.rotate(np.identity(3), rm, np.array([0,0,0]),
                             0, self.draw.gSize[i], i)
        self.pendingShader = True
        
    def rotateLight(self):
        a = self.directionalLights[0]["dir"][1] - 0.1*pi
        a += 0.05
        d = 0.1*pi + (a % (pi*0.9))
        self.directionalLights[0]["dir"][1] = d
        ti = abs(self.directionalLights[0]["dir"][1])

        sc = self.shadowCams[0]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -40 * viewVec(*sc["dir"]) + numpy.array([0, 5, 0])
        self.updateShadowCam(0)
        sc["bias"] = 0.1 * abs(cos(ti)) + 0.02
        self.shadowMap(0, bias=sc["bias"])
        self.pendingShader = True
        
    def createObjects(self):
        st = time.time()
        print("creating objects")

        mpath = "../Models/"

        self.addVertObject(VertTerrain, [-30, -4, -30],
                           heights="Assets/Landscape2.png",
                           texture="Assets/Grass1.png",
                           scale=(0.375, 5, 0.375),
                           shadow="CR", mip=1, uvspread=2)
        self.terrain = self.vertObjects[-1]
        
        self.directionalLights.append({"dir":[pi/2, 2.14], "i":1.1})

        self.shadowCams.append({"pos":[20, 5, 20], "dir":[pi/2, 1.2],
                                "size":4096, "scale":64, "angle":2,
                                "light":self.directionalLights[-1],
                                "mode":"ortho",
                                "softdist":4,
                                "bias":0.2})
        
        self.makeObjects()
        self.skyBox = TexSkyBox(self, self, 12, "Skyboxes/Skybox1a.png")
        self.skyBox.created()

        print("done in", time.time() - st, "s")

    def postProcess(self):
        self.draw.gamma()
        
    def onMove(self):
        pass
    def onStart(self):
        self.shadowObjects()
        self.setupShadowCams()
        self.rotateLight()
        if self.fopen is not None:
            self.importChar(self.fopen)

    def initRig(self, fn):
        r = json.load(open(fn))
        self.rig = Rig(r)
        self.allBones = self.rig.allBones
        self.b0 = self.rig.b0
        
    def frameUpdate(self):
        pass

    def exportChar(self, fn):
        try: a = self.testv
        except AttributeError: return
        mtln = ".".join(fn.split(".")[:-1]) + ".mtl"
        for i in self.ctexn:
            self.vertBones[i] = self.draw.getBoneWeights(i)
            self.vertPoints[i] = self.draw.getVertPoints(i)[:,:,:3]
            self.vertNorms[i] = self.draw.getVertNorms(i)[:,:,:3]
        from ObjExportAnim import exportObjMultiTexNormBones
        #k = list(self.vtNames.keys())
        texs = []
        c = self.testv
        texs.append(c.mtlTexFname)
        while c.nextMtl is not None:
            c = c.nextMtl
            texs.append(c.mtlTexFname)
        try:
            exportObjMultiTexNormBones(fn, mtln, texs,#[k[i] for i in self.ctexn],
                                   [self.vertPoints[i] for i in self.ctexn],
                                   [np.stack((self.vertU[i],
                                              self.vertV[i]), axis=2)
                                    for i in self.ctexn],
                                   [self.vertNorms[i] for i in self.ctexn],
                                   [self.vertBones[i] for i in self.ctexn],
                                   mtlAlias=mtln.split("/")[-1])
        except FileNotFoundError:
            return
        rn = ".".join(fn.split(".")[:-1]) + ".rig"
        self.exportRig(rn)

    def exportRig(self, fn):
        self.b0.N = len(self.allBones)
        with open(fn, "w") as fo:
            json.dump(self.b0.exportRig(), fo)

def run(f=None):
    w = 960; h = 640
    try:
        with open("Settings.txt") as sf:
            for line in sf:
                if line[:2] == "W=": w=int(line[2:])
                if line[:2] == "H=": h=int(line[2:])
    except FileNotFoundError:
        with open("Settings.txt", "w") as f:
            f.write("Settings for AXI Visualizer\n\nCL=0:0\nW=960\nH=640")
    
    app = ThreeDApp(w, h, fopen=f)
    app.start()
    app.runBackend()
    app.finish()

if __name__ == "__main__":
    run()
