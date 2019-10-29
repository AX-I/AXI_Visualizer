# AXI Visualizer: Animator
# Skeletal animation
# Rig posing

# New: save project

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

from Rig import Rig

class ThreeDApp(ThreeDBackend):
    def __init__(self, width=960, height=640, fovx=80, fopen=None):
        super().__init__(width, height, fovx=fovx)

        self.fopen = fopen

        self.changeTitle("AXI Visualizer: Animator")

        self.α = -0.78
        self.β = 0.52
        self.pos = numpy.array([27.2, 5.2, 8.1])
        
        self.camSpeed = 0.1

        self.mouseSensitivity = 20

        self.ambLight = 0.1

        self.sphereRot = [0,0,0]

        self.bi = 0
        self.ctexn = []
        self.cpropn = {}

        self.poset = 0
        self.pstep = 0.04

        self.hSwitch = time.time()
        self.hi = False
        self.doFlash = True

        self.drawAxes = True
        self.maxFPS = 33

        self.guiType = "Pose"

        self.da = 5

    def moveKey(self, key):
        if key == "u":   self.svert = 1
        elif key == "d": self.svert = -1
        elif key == "r": self.shorz = 1
        elif key == "l": self.shorz = -1
        elif key == "ZV": self.svert = 0
        elif key == "ZH": self.shorz = 0
        elif key[:4] == "save": self.exportChar(key[4:])
        elif key[:4] == "open": self.importChar(key[4:])
        elif key[:5] == "psave": self.savePose(key[5:])
        elif key[:5] == "popen": self.loadPose(key[5:])
        elif key[:6] == "switch": self.switchBone(key[6:])
        elif key[:5] == "oopen": self.addProp(key[5:])
        elif key[:5] == "scale": self.scaleProp(key[5:])
        elif key[:7] == "pswitch": self.switchProp(key[7:])
        elif key[:5] == "asave": self.saveAll(key[5:])
        elif key[:5] == "rsave": self.saveProj(key[5:])
        elif key[:5] == "ropen": self.loadProj(key[5:])

    def customizeFrontend(self):
        self.bindKey("p", "pP"); self.makeHandler("pP", self.rotateThing)
        self.bindKey("0", "r0"); self.makeHandler("r0", self.resetRot)

        self.bindKey("r", "rR"); self.makeHandler("rR", self.rotateLight)

        self.bindKey("4", "ax"); self.makeHandler("ax", self.tgAxes)
        self.bindKey("5", "f"); self.makeHandler("f", self.tgFlash)

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
                              filename=fn,
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

        if not self.testv.animated:
            root = Tk()
            root.withdraw()
            showerror("Error", "Model does not have animation data.")
            root.destroy()
            return

        try:
            self.initRig(".".join(fn.split(".")[:-1]) + ".rig")
        except FileNotFoundError:
            root = Tk()
            root.withdraw()
            showerror("Error", "No rig found for model.")
            root.destroy()
            return
        
        try:
            self.createInserted(reversed(ctexn))
        except Exception as e:
            root = Tk()
            root.withdraw()
            showerror("Error", "Couldn't load model:\n"+traceback.format_exc())
            root.destroy()
            return

        self.allBones = self.rig.allBones
        for i in range(len(self.vertBones)):
            if len(self.vertBones[i]) > 0:
                self.draw.addBoneWeights(i, self.vertBones[i])

        for b in range(len(self.rig.allBones)):
            for i in range(len(self.vertBones)):
                vb = self.vertBones[i]
                if len(vb) > 0:
                    self.draw.initBoneOrigin(self.rig.allBones[b].origin, b, i)

        self.updateR()

        self.cb = self.allBones[self.bi]

        self.bindKey("f", "fF"); self.makeHandler("fF", self.chooseBone)
        self.bindKey("F", "fg"); self.makeHandler("fg", self.backBone)

        self.bindKey("1", "xp"); self.makeHandler("xp", self.xp)
        self.bindKey("2", "yp"); self.makeHandler("yp", self.yp)
        self.bindKey("3", "zp"); self.makeHandler("zp", self.zp)
        self.bindKey("!", "xn"); self.makeHandler("xn", self.xn)
        self.bindKey("@", "yn"); self.makeHandler("yn", self.yn)
        self.bindKey("#", "zn"); self.makeHandler("zn", self.zn)
        self.bindKey("+", "ap"); self.makeHandler("ap", self.ap)
        self.bindKey("-", "am"); self.makeHandler("am", self.am)
        
        self.bindKey("t", "tT"); self.makeHandler("tT", self.select)
        
        self.getText()
        self.P.put(("rnum", self.rigText, self.bi))

        self.pendingShader = True
        self.rotateLight()

        self.focus = "Char"

    def initRig(self, fn):
        r = json.load(open(fn))
        self.rig = Rig(r)
        #self.rig.b0.offset = np.array((0,0,0,1.))
        self.rig.b0.offset = np.array((26,3.6,10,1))
        self.rig.b0.updateTM()
        self.cb = self.rig.b0

        self.draw.initBoneTransforms("a", len(self.rig.allBones))

    def saveProj(self, fn):
        if fn == "": return
        out = {"char":self.testv.filename, "objs":[]}
        try: out["pose"] = self.poseFile
        except AttributeError: pass
        for v in self.vertObjects:
            if not v.animated:
                try:
                    if v.root:
                        a = {"filename":v.filename,
                             "coords":list(v.coords),
                             "rm":v.rotMat.tolist(),
                             "scale":v.scale.tolist()}
                        out["objs"].append(a)
                except AttributeError:
                    if type(v) is VertModel:
                        raise
        with open(fn, "w") as f:
            json.dump(out, f)
        
    def loadProj(self, fn):
        if fn == "": return
        with open(fn) as f:
            x = json.load(f)
        
        self.importChar(x["char"])
        
        for v in x["objs"]:
            self.addProp(v["filename"])
            self.scaleProp(v["scale"])
            self.rotProp1(np.array(v["rm"]))
            #self.rotProp1(v["angles"])
            for i in self.cpropn[self.curPN]:
                self.draw.translate(np.array(v["coords"]),
                                    self.vertObjects[i].cStart*3,
                                    self.vertObjects[i].cEnd*3, i)
            self.curProp.coords = np.array(v["coords"])

        if "pose" in x: self.loadPose(x["pose"])
        
    def addProp(self, fn):
        if fn == "": return
        try:
            self.insertObject(VertModel, [0,0,0],
                              filename=fn, animated=False,
                              scale=1, shadow="C")
        except Exception as e:
            root = Tk()
            root.withdraw()
            showerror("Error", "Couldn't load model:\n"+traceback.format_exc())
            root.destroy()
            return
        ctexn = []
        c = self.vertObjects[-1]
        ctexn.append(c.texNum)
        while c.nextMtl is not None:
            c = c.nextMtl
            ctexn.append(c.texNum)
        self.cpropn[fn.split("/")[-1]] = ctexn

        try:
            self.createInserted(reversed(ctexn))
        except Exception as e:
            root = Tk()
            root.withdraw()
            showerror("Error", "Couldn't load model:\n"+traceback.format_exc())
            root.destroy()
            for i in ctexn:
                del self.vertPoints[-1], self.vertNorms[-1]
                del self.vertU[-1], self.vertV[-1]#, self.vertLight[-1]
                del self.vtextures[-1], self.texUseAlpha[-1]
                del self.vertObjects[-1]
            return
        
        self.pendingShader = True
        self.rotateLight()
        
        self.P.put(("pnum", list(self.cpropn)))
        self.curProp = self.vertObjects[-1]
        self.curPN = fn.split("/")[-1]
        
    def switchProp(self, pn):
        self.curProp = self.vertObjects[self.cpropn[pn][0]]
        self.curPN = pn

        self.makeHandler("xp", lambda a=0, b=1: self.rotProp(a,b))
        self.makeHandler("yp", lambda a=1, b=1: self.rotProp(a,b))
        self.makeHandler("zp", lambda a=2, b=1: self.rotProp(a,b))
        self.makeHandler("xn", lambda a=0, b=-1: self.rotProp(a,b))
        self.makeHandler("yn", lambda a=1, b=-1: self.rotProp(a,b))
        self.makeHandler("zn", lambda a=2, b=-1: self.rotProp(a,b))

        self.focus = "Prop"
        
    def scaleProp(self, s):
        try: a = self.curPN
        except: return
        try: s = float(s)
        except: return
        for i in self.cpropn[self.curPN]:
            self.draw.scale(self.curProp.coords, s, i)
        self.curProp.scale = self.curProp.scale * s
    def rotProp(self, ax, sig):
        r = [0,0,0]
        r[ax] = sig * self.da*pi/180
        rm = rotMat(r)
        for i in self.cpropn[self.curPN]:
            self.draw.rotate(np.identity(3), rm, self.curProp.coords,
                             0, self.draw.gSize[i], i)
        #self.curProp.angles += np.array(r, dtype="float")
        self.curProp.rotMat = self.curProp.rotMat @ rm
    def rotProp1(self, rm):
        #rm = rotMat(r)
        for i in self.cpropn[self.curPN]:
            self.draw.rotate(np.identity(3), rm, self.curProp.coords,
                             0, self.draw.gSize[i], i)
        #self.curProp.angles += np.array(r, dtype="float")
        self.curProp.rotMat = self.curProp.rotMat @ rm

    def select(self):
        self.selecting = not self.selecting

    def cSelect(self, commit, xy):
        if self.focus == "Prop":
            cz = (self.curProp.coords - self.pos) @ self.vv
            xy -= np.array((self.W2, self.H2))
            p = self.pos + cz*self.viewVec()
            p += cz* -xy[0]/self.scale * self.vVhorz()
            p += cz* xy[1]/self.scale * self.vVvert()

            self.curProp.oldCoords = self.curProp.coords
            self.curProp.coords = p
            for i in self.cpropn[self.curPN]:
                self.draw.translate(p - self.curProp.oldCoords,
                                    self.vertObjects[i].cStart*3,
                                    self.vertObjects[i].cEnd*3, i)
            
            self.axPoints = self.baseAxPoints + p
        elif self.focus == "Char":
            cz = (self.rig.b0.offset[:3] - self.pos) @ self.vv
            xy -= np.array((self.W2, self.H2))
            p = self.pos + cz*self.viewVec()
            p += cz* -xy[0]/self.scale * self.vVhorz()
            p += cz* xy[1]/self.scale * self.vVvert()

            self.rig.b0.offset = np.array((*p, 1))
            self.rig.b0.updateTM()
            self.updateR()
            
    def tgAxes(self):
        self.drawAxes = not self.drawAxes
    def tgFlash(self):
        self.doFlash = not self.doFlash
        
    def loadPose(self, fn):
        try: a = self.rig
        except AttributeError: return
        try:
            self.p = json.load(open(fn))
        except FileNotFoundError:
            return
        self.rig.importPose(self.p)
        self.updateR()
        self.poseFile = fn
        
    def savePose(self, f):
        try: a = self.rig
        except AttributeError: return
        with open(f, "w") as fo:
            json.dump(self.rig.exportPose(), fo)

    def ap(self):
        self.da *= 2
    def am(self):
        self.da /= 2
    def xp(self):
        self.cb.angles[0] += self.da*pi/180; self.cb.rotate()
        self.updateR()
    def yp(self):
        self.cb.angles[1] += self.da*pi/180; self.cb.rotate()
        self.updateR()
    def zp(self):
        self.cb.angles[2] += self.da*pi/180; self.cb.rotate()
        self.updateR()
    def xn(self):
        self.cb.angles[0] -= self.da*pi/180; self.cb.rotate()
        self.updateR()
    def yn(self):
        self.cb.angles[1] -= self.da*pi/180; self.cb.rotate()
        self.updateR()
    def zn(self):
        self.cb.angles[2] -= self.da*pi/180; self.cb.rotate()
        self.updateR()

    def updateR(self):
        self.axPoints = self.rig.allBones[self.bi].getPoints(self.baseAxPoints)
        self.updateRig(self.rig, range(len(self.vtNames)), "a")
        self.P.put(("rots", self.rig.allBones[self.bi].angles))

    def chooseBone(self):
        self.bi += 1
        if self.bi >= len(self.rig.allBones):
            self.bi = 0
        self.cb = self.rig.allBones[self.bi]
        self.axPoints = self.cb.getPoints(self.baseAxPoints)
        self.pendingShader = True
        self.getText()
        self.P.put(("rnum", self.rigText, self.bi))
        self.P.put(("rots", self.rig.allBones[self.bi].angles))
    def backBone(self):
        self.bi -= 2
        self.chooseBone()
    def switchBone(self, n):
        self.bi = int(n) - 1
        self.chooseBone()
        self.makeHandler("xp", self.xp)
        self.makeHandler("yp", self.yp)
        self.makeHandler("zp", self.zp)
        self.makeHandler("xn", self.xn)
        self.makeHandler("yn", self.yn)
        self.makeHandler("zn", self.zn)
        self.focus = "Char"
        
    def rotateThing(self):
        self.sphereRot[1] += 0.1
        self.rig.b0.rotate(self.sphereRot)
        self.updateR()
        
        self.shadowObjects()
        self.pendingShader = True
    
    def resetRot(self):
        cb = self.rig.allBones[self.bi]
        cb.rotate([0,0,0])
        self.updateR()
        
        self.shadowObjects()
        self.pendingShader = True
        
    def rotateLight(self):
        #self.directionalLights[0]["dir"][0] += 0.01
        a = self.directionalLights[0]["dir"][1] - 0.1*pi
        a += 0.05
        d = 0.1*pi + (a % (pi*0.9))
        self.directionalLights[0]["dir"][1] = d
        self.directionalLights[1]["dir"][1] = d + pi
        ti = abs(self.directionalLights[0]["dir"][1])
        
        sc = self.shadowCams[0]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -40 * viewVec(*sc["dir"]) + numpy.array([20, 5, 20])
        self.updateShadowCam(0)
        sc["bias"] = 0.1 * abs(cos(ti)) + 0.02
        self.shadowMap(0, bias=sc["bias"])
        self.pendingShader = True
        
    def createObjects(self):
        self.addVertObject(VertTerrain, [-10, 0, -10],
                           heights="Assets/Landscape2.png",
                           texture="Assets/Grass1.png",
                           scale=(0.375, 5, 0.375),
                           shadow="CR", mip=1, uvspread=2)
        
        self.directionalLights.append({"dir":[pi/2, 2.4], "i":1.1})
        self.directionalLights.append({"dir":[pi/2, 2.4+pi], "i":0.3})

        self.shadowCams.append({"pos":[20, 5, 15], "dir":[pi/2, 1.2],
                                "size":2048, "scale":48, "angle":2,
                                "light":self.directionalLights[-1],
                                "mode":"ortho",
                                "softdist":4,
                                "bias":0.2})
        
        self.makeObjects()
        self.skyBox = TexSkyBox(self, self, 12, "Skyboxes/Skybox1a.png")
        self.skyBox.created()

    def postProcess(self):
        self.draw.gamma()
                
    def onStart(self):
        self.shadowObjects()
        self.setupShadowCams()
        self.rotateLight()
        if self.fopen is not None:
            self.importChar(self.fopen)
        
    def frameUpdate(self):
        if self.doFlash and (time.time() - self.hSwitch > 0.4):
            self.hSwitch = time.time()
            self.hi = not self.hi
            h = 3 if self.hi else 0.1
            for i in self.ctexn:
                if len(self.vertBones[i]) > 0:
                    self.draw.highlight(self.bi, numpy.array([0,0,0]), 0,
                                        h, h, h, 0, i, 0)

    def exportChar(self, fn):
        try: a = self.testv
        except AttributeError: return
        
        tmpo = self.rig.b0.offset
        self.rig.b0.offset = np.array((0,0,0,1.))
        self.rig.b0.updateTM()
        self.updateR()
        
        mtln = ".".join(fn.split(".")[:-1]) + ".mtl"
        for i in self.ctexn:
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

        self.rig.b0.offset = tmpo
        self.rig.b0.updateTM()
        self.updateR()

    def saveAll(self, fn):
        mtln = ".".join(fn.split(".")[:-1]) + ".mtl"
        for i in range(len(self.vertPoints)):
            self.vertPoints[i] = self.draw.getVertPoints(i)[:,:,:3]
            self.vertNorms[i] = self.draw.getVertNorms(i)[:,:,:3]
        from ObjExport import exportObjMultiTexNorm
        #k = list(self.vtNames.keys())
        texs = []
        for i in self.vertObjects:
            try: texs.append(i.mtlTexFname)
            except AttributeError: texs.append(i.texName)
        objAll = range(len(self.vertU))
        try:
            exportObjMultiTexNorm(fn, mtln, texs,
                                  self.vertPoints,
                                  [np.stack((self.vertU[i],
                                             self.vertV[i]), axis=2)
                                   for i in objAll],
                                  self.vertNorms,
                                  mtlAlias=mtln.split("/")[-1])
        except FileNotFoundError:
            return

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
    app.fps()

if __name__ == "__main__":
    run()
