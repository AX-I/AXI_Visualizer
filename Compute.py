# 3D render
# Backend process

"""
Currently working on hot loading, gui tools, complete package

General:
- landscape editing / interactivity
- lightmaps, per-object lighting
- procedural sky
- volumetric light & shadow
- physics integration

- brushes / procedural placement
- auto LOD
- billboards
- line renderer
- full UI
- profiling
"""

import multiprocessing as mp
from queue import Empty, Full

from math import sin, cos, pi
import numpy as np
import numexpr as ne
import time
from Utils import *
import Ops3 as Ops

import Visualizer as VS

from TexObjects import TexSkyBox
from Cubemap import CubeMap
from VertObjects import VertSphere, VertModel, VertTerrain, VertPlane, VertWater

import sys, os
from PIL import Image

class ThreeDBackend:
    def __init__(self, width, height,
                 scale=600, fovx=None,
                 downSample=1):
        pipe = rec = mp.Queue(3)
        
        self.evtQ = mp.Queue(64)
        self.infQ = mp.Queue(10)
        
        self.P = pipe
        self.recP = rec
        self.handles = {}
        self.full = 0
        self.empty = 0
        
        self.W = width
        self.H = height
        self.downSample = downSample

        self.α = 0
        self.β = 0

        self.estWedges = 0
        self.actWedges = 0
        self.vertObjects = []

        self.vtNames = {}
        self.vertPoints = []
        self.vertNorms = []
        self.vertU = []
        self.vertV = []
        self.vtextures = []

        self.vertBones = []

        self.texAlphas = []
        self.texUseAlpha = []
        self.vaNames = {}
        self.texShadow = []
        self.texMip = []
        self.texRefl = []
        self.texReflNames = {}
        
        self.skyPoints = []
        self.skyAvgPos = []
        self.skyU = []
        self.skyV = []
        self.skyTex = None
        
        self.W2 = int(self.W/2)
        self.H2 = int(self.H/2)

        self.pos = np.array([0., 0., 0.])
        self.vc = self.viewCoords()

        self.camSpeed = 0.3
        self.speed = np.array([0., 0., 0.])
        self.svert = 0
        self.shorz = 0
        self.panning = False
        self.mouseSensitivity = 50

        if fovx is not None:
            self.scale = self.W2 / np.tan(fovx*pi/360)
        else:
            self.scale = scale
        self.cullAngle = self.scale / np.sqrt(self.scale**2 + self.W2**2 + self.H2**2) * 0.85
        self.cullAngleX = self.W2 / self.scale + 2
        self.cullAngleY = self.H2 / self.scale + 2
        self.fovX = np.arctan(self.W2 / self.scale) * 360/pi
        self.fovY = np.arctan(self.H2 / self.scale) * 360/pi

        self.directionalLights = []

        self.pointLights = []
        self.spotLights = []
        self.shadowCams = []
        self.ambLight = 0.1

        self.drawCross = False
        self.drawAxes = True
        self.axPoints = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                                 dtype="float")*0.25
        self.baseAxPoints = np.array(self.axPoints)

        self.drawLines = False
        self.linePoints = np.array([])
        
        self.timfps = np.zeros(12)
        self.numfps = 0
        self.maxFPS = 60

        self.selecting = False
        self.genNewBones = False

        self.guiType = None

        if not os.path.isdir("Screenshots"):
            os.mkdir("Screenshots")

    def start(self):
        self.createObjects()

        self.vertPoints = [np.array(i) for i in self.vertPoints]
        self.vertNorms = [np.array(i) for i in self.vertNorms]
        self.vertU = [np.array(i) for i in self.vertU]
        self.vertV = [np.array(i) for i in self.vertV]
        self.vertLight = [np.ones((i.shape[0], 3)) for i in self.vertPoints]
        
        self.skyPoints = np.array(self.skyPoints)
        self.skyAvgPos = np.array(self.skyAvgPos).T
        self.skyU = np.array(self.skyU) / 6
        self.skyV = np.array(self.skyV)
        self.nWS = self.skyAvgPos.shape[1]
        
        #Luv = len(self.vertU)
        Luv = 64
        
        self.draw = Ops.CLDraw(self.nWS//2, self.skyTex.shape[0],
                               Luv, self.W, self.H)

        #self.draw.ambLight = np.float32(self.ambLight)

        self.draw.setScaleCull(self.scale, self.cullAngleX, self.cullAngleY)

        self.skyTex = np.array(self.skyTex)
        self.draw.setSkyTex(self.skyTex[:,:,0],
                            self.skyTex[:,:,1],
                            self.skyTex[:,:,2],
                            self.skyTex.shape[1])
            
        for i in range(len(self.vtextures)):
            tex = self.vtextures[i]
            if self.texMip[i]:
                t = createMips(tex)
                self.draw.addTextureGroup(
                    self.vertPoints[i].reshape((-1,3)),
                    np.stack((self.vertU[i], self.vertV[i]), axis=2).reshape((-1,3,2)),
                    self.vertNorms[i].reshape((-1,3)),
                    t[:,0], t[:,1], t[:,2],
                    mip=tex.shape[0])
            else:
                self.draw.addTextureGroup(
                    self.vertPoints[i].reshape((-1,3)),
                    np.stack((self.vertU[i], self.vertV[i]), axis=2).reshape((-1,3,2)),
                    self.vertNorms[i].reshape((-1,3)),
                    tex[:,:,0], tex[:,:,1], tex[:,:,2])

        for tex in self.texAlphas:
            self.draw.addTexAlpha(tex)

        if self.genNewBones:
            self.vertBones = []
            for i in self.vertLight:
                self.vertBones.append(np.zeros((i.shape[0], 3), dtype="int"))
        
        for i in range(len(self.vertBones)):
            if len(self.vertBones[i]) > 0:
                self.draw.addBoneWeights(i, self.vertBones[i])
        
        bargs = (self.recP, self.evtQ, self.infQ,
                 self.W, self.H, self.mouseSensitivity,
                 self.downSample, self.guiType)
        self.frontend = mp.Process(target=VS.runGUI, args=bargs)

        self.frontend.start()

        self.customizeFrontend()

    def addVertObject(self, objClass, *args, **kwargs):
        thing = objClass(self, *args, **kwargs)
        self.estWedges += thing.estWedges
        self.vertObjects.append(thing)
        return True
    
    def insertObject(self, objClass, *args, **kwargs):
        thing = objClass(self, *args, **kwargs, late=1) #, newTexName=1)
        self.vertObjects.append(thing)

    def createInserted(self, ctexn):
        for i in ctexn:
            thing = self.vertObjects[i]
            thing.created(False)
            self.vertLight.append(np.ones((self.vertPoints[i].shape[0], 3)))
            tex = self.vtextures[i]
            self.draw.addTextureGroup(
                self.vertPoints[i].reshape((-1,3)),
                np.stack((self.vertU[i], self.vertV[i]), axis=2).reshape((-1,3,2)),
                self.vertNorms[i].reshape((-1,3)),
                tex[:,:,0], tex[:,:,1], tex[:,:,2])

            if thing.useAlpha:
                self.draw.addTexAlpha(self.texAlphas[thing.useAlpha-1])

        self.shadowObjects()

    def updateRig(self, rig, ct, name, offset=0):
        bt = rig.b0.getTransform()
        self.draw.setBoneTransform(name, bt)
        for i in ct:
            if len(self.vertBones[i]) > 0:
                self.draw.boneTransform(0, self.vertPoints[i].shape[0]*3, i, name,
                                        offset)

        self.pendingShader = True

    def createObjects(self):
        pass
    def profileStart(self):
        self.profileTime = time.time()
        self.numfps += 1
    def renderProfile(self, n):
        self.timfps[n] += time.time() - self.profileTime

    def makeObjects(self, v=1):
        print("estimated # of wedges:", self.estWedges)
        self.actWedges = 0
        for o in self.vertObjects:
            o.created()
            self.actWedges += o.numWedges
            if v:
                print("\r", int(100 * self.actWedges / self.estWedges), "%",
                      sep="", end="")
        print("\nactual # of wedges:", self.actWedges)
    
    def addVertObject(self, objClass, *args, **kwargs):
        thing = objClass(self, *args, **kwargs)
        self.estWedges += thing.estWedges
        self.vertObjects.append(thing)
        return True
        
    def render(self):
        self.renderProfile(1)
        
        self.renderProfile(2)
        
        self.vc = self.viewCoords()
        self.vMat = np.stack((self.vv,self.vVhorz(),self.vVvert()))
        self.draw.setPos(self.vc)
        self.draw.setVM(self.vMat)
        
        if len(self.skyPoints) > 0:
            skVisible = (self.vv @ self.skyAvgPos) > self.cullAngle
            
            tempSP = np.concatenate(self.skyPoints[skVisible], axis=0)
            tempU = np.concatenate(self.skyU[skVisible], axis=0)
            tempV = np.concatenate(self.skyV[skVisible], axis=0)
            skyproj = self.skyProject(tempSP)

            self.draw.skydraw(skyproj, tempU, tempV)
        
        self.renderProfile(3)
        
        self.draw.clearZBuffer()

        self.renderProfile(4)
        if len(self.shadowCams) == 1:
            s = [0,0]
        elif len(self.shadowCams) == 2:
            s = [0,1]
        self.draw.drawAll(self.texUseAlpha, self.texShadow,
                          self.texMip, self.texRefl, shadowIds=s)
        
        self.renderProfile(5)
        
        a = None
        if self.drawAxes:
            if (((self.axPoints - self.pos) @ self.vv) > 0).all():
                a = self.projectPoints(self.axPoints).astype("int")

        b = None
        if self.drawLines:
            vp = ((self.linePoints - self.pos) @ self.vv) > 0
            for i in range(len(vp)//2):
                if not (vp[2*i] & vp[2*i+1]): vp[2*i] = False; vp[2*i+1] = False
            visLP = self.linePoints[vp]
            b = self.projectPoints(visLP).astype("int")

        self.renderProfile(6)
        self.renderProfile(7)

        self.postProcess()
        
        self.renderProfile(8)
        #aa = Image.fromarray((rgb>>8).astype("uint8"), "RGB")
        #aa.save("test.png")

        result = self.draw.getFrame()
        self.rgb = np.stack(result[:3], axis=2)
        self.zb = result[3]
        
        self.renderProfile(9)
        
        return [self.rgb, a, self.selecting, self.drawCross, b]

    def projectPoints(self, xyzn):
        dxy = (xyzn - self.pos) @ self.vMat.T

        d = 1 / dxy[:,0]
        x = dxy[:,1]
        y = dxy[:,2]
        s = self.scale
        dw, dh = self.W2, self.H2
        x2d = ne.evaluate("x * d * -s + dw")
        y2d = ne.evaluate("y * d * s + dh")
        xy2d = np.stack((x2d, y2d), axis=1)
        return xy2d
        
    def simpleShaderVert(self):
        lightI = np.array([d["i"] for d in self.directionalLights])
        lightD = np.array([viewVec(*d["dir"]) for d in self.directionalLights])
        self.draw.vertLight(lightI, lightD)
           
    def shadowObjects(self):
        sobj = np.full((len(self.vertU),), False)
        for o in self.vertObjects:
            if o.castShadow:
                sobj[o.texNum] = True
        
        self.castObjs = sobj

    def setupShadowCams(self):
        for i in range(len(self.shadowCams)):
            s = self.shadowCams[i]
            self.draw.addShadowMap(i, s["size"], s["scale"],
                                   self.ambLight)

    def updateShadowCam(self, i):
        s = self.shadowCams[i]
        self.draw.placeShadowMap(i, s["pos"], s["dir"], self.ambLight)

    def shadowMap(self, i, castObjs=None, bias=0.2):
        if castObjs is None:
            castObjs = self.castObjs
        if len(self.shadowCams) == 0:
            print("No shadow cams")
            return False
        sc = self.shadowCams[i]
        
        self.draw.clearShadowMap(i)

        self.draw.shadowMap(i, castObjs, self.texUseAlpha, bias)
        
    def viewCoords(self):
        return self.pos

    def viewVec(self):
        v = np.array([sin(self.α) * cos(self.β),
                      -sin(self.β),
                      cos(self.α) * cos(self.β)])
        return v
    def vVvert(self):
        b2 = self.β - pi/2
        v = np.array([sin(self.α) * cos(b2),
                      -sin(b2),
                      cos(self.α) * cos(b2)])
        return -v
    def vVhorz(self):
        a2 = self.α + pi/2
        v = np.array([sin(a2), 0, cos(a2)])
        return -v

    def skyProject(self, xyzn):
        vn = np.array(xyzn)
        pn = np.stack((self.vv, self.vVhorz(), self.vVvert())).T
        dxy = vn @ pn

        d = dxy[:,0]
        x = dxy[:,1]
        y = dxy[:,2]
        s = self.scale
        dw = self.W2
        dh = self.H2
        x2d = ne.evaluate("x / d * -s + dw")
        y2d = ne.evaluate("y / d * s + dh")

        xy2d = np.stack((x2d, y2d), axis=1)
        return xy2d

    def doEvent(self, action):
        self.α += action[1]
        self.β += action[2]
        
    def moveKey(self, key):
        if key == "u":   self.svert = 1
        elif key == "d": self.svert = -1
        elif key == "r": self.shorz = 1
        elif key == "l": self.shorz = -1
        elif key == "ZV": self.svert = 0
        elif key == "ZH": self.shorz = 0
            
    def pan(self, d):
        self.panning = True
        dx = d[0]
        dy = d[1]
        self.pos += np.array([
            -(dx * cos(self.α) - dy * sin(self.β) * sin(self.α)),
            dy * cos(self.β),
            dx * sin(self.α) + dy * sin(self.β) * cos(self.α)])

    def runBackend(self):
        self.doQuit = False
        self.pendingShader = True
        frontReady = False

        print("waiting for frontend", end="")
        while (not frontReady):
            try:
                if self.evtQ.get(True, 1) == ["ready"]:
                    frontReady = True
            except Empty:
                print(".", end="")
        print("\nstarting render")

        self.startTime = time.time()
        self.frameNum = 0
        self.totTime = 0

        self.onStart()
        
        while (not self.doQuit):
            self.profileStart()

            self.vv = self.viewVec()

            self.speed[:] = self.svert * self.vv
            self.speed += self.shorz * -self.vVhorz()
            self.pos += self.camSpeed * self.speed
            if (self.speed != 0).any() or self.panning:
                self.panning = False

            self.frameUpdate()
            self.vv = self.viewVec()
            
            self.renderProfile(0)
            
            if self.pendingShader:
                self.simpleShaderVert()
                self.pendingShader = False
            
            r = self.render()
            data = ("render", np.array(r, dtype="object"))
            try:
                self.P.put_nowait(data)
            except Full:
                self.full += 1
            self.renderProfile(10)
            
            while not self.evtQ.empty():
                self.processEvent()
                    
            self.renderProfile(11)

            self.frameNum += 1
            dt = time.time() - self.startTime
            if dt < (1/self.maxFPS):
                time.sleep((1/self.maxFPS) - dt)
            dt = time.time() - self.startTime
            self.totTime += dt
            self.startTime = time.time()

        try:
            self.P.put(None, True, 1)
        except (Full, BrokenPipeError):
            pass

    def processEvent(self):
        try:
            action = self.evtQ.get_nowait()
        except Empty:
            self.empty += 1
        else:
            if action is None:
                self.doQuit = True
            elif action[0] == "event":
                self.doEvent(action)
            elif action[0] == "eventk":
                self.moveKey(action[1])
            elif action[0] == "eventp":
                self.pan(action[1])
            elif action[0] == "select":
                self.cSelect(*action[1])
            elif action in self.handles:
                self.handles[action]()

    def finish(self):
        try:
            while not self.P.empty():
                self.P.get(True, 0.5)
        except: pass
        try:
            while not self.evtQ.empty():
                self.evtQ.get(True, 0.5)
        except: pass
        
        self.P.close()
        self.evtQ.close()
        time.sleep(1)
        print("closing", end="")
        self.P.join_thread()
        self.evtQ.join_thread()
        while mp.active_children():
            time.sleep(0.5)
            print(".", end="")
        self.frontend.join()
        time.sleep(0.5)
        print("\nclosed frontend")

    def changeTitle(self, t):
        self.P.put(("title", str(t)))
    def bindKey(self, k, mess):
        self.P.put(("key", k, mess))
    def makeHandler(self, e, f):
        self.handles[e] = f

    def frameUpdate(self):
        pass
    def postProcess(self):
        pass
    def onMove(self):
        pass
    def facing(self):
        return (self.α, self.β)

    def fps(self):
        print("fps:", self.frameNum / self.totTime)

    def printProfile(self):
        st = "full:" + str(self.full) + ", empty:" + str(self.empty)
        st += "\n" + str(self.frameNum) + " frames"
        st += "\nAv. fps: " + str(np.reciprocal(self.timfps) * self.numfps)
        a = self.timfps / self.numfps
        st += "\nAv. spf:" + str(a)
        st += "\n" + str(a[0])
        for i in range(len(a) - 1):
            st += "\n" + str(a[i + 1] - a[i])
        st += "\noverall spf: " + str(self.totTime / self.frameNum)
        return st

if __name__ == "__main__":
    print("Please subclass this in a separate file!")
