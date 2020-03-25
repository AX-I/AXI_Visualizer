# 3D render
# Backend process

"""
Hierarchical tiled scanline rasterization
"""

import multiprocessing as mp
from queue import Empty, Full

from math import sin, cos, pi, ceil, atan2
import numpy as np

import time
from Utils import *

import Visualizer as VS

from TexObjects import TexSkyBox, TexSkySphere, TexSkyHemiSphere, Sun
from Cubemap import CubeMap
from VertObjects import VertSphere, VertModel, VertTerrain, VertPlane, VertWater

from ParticleSystem import ContinuousParticleSystem, SpiralParticleSystem
import sys, os
from PIL import Image, PngImagePlugin
import json

def getTexture(fn):
    ti = Image.open(fn).convert("RGB")
    if ti.size[0] != ti.size[1]:
        raise ValueError("Texture is not square!")
    if (ti.size[0] & (ti.size[0] - 1)) != 0:
        #print("Texture is not a power of 2, resizing up.")
        n = 2**ceil(log2(ti.size[0]))
        ti = ti.resize((n,n))
    ta = np.array(ti).astype("float")
    ta *= 64 * 4
    np.clip(ta, None, 256*256-1, ta)
    return np.array(ta.astype("uint16").reshape((-1,3)))

def getTexture1(fn):
    """keep 2d"""
    ti = Image.open(fn).convert("RGB")
    if ti.size[0] != ti.size[1]:
        raise ValueError("Texture is not square!")
    if (ti.size[0] & (ti.size[0] - 1)) != 0:
        #print("Texture is not a power of 2, resizing up.")
        n = 2**ceil(log2(ti.size[0]))
        ti = ti.resize((n,n))
    ta = np.array(ti).astype("float")
    ta *= 64 * 4
    np.clip(ta, None, 256*256-1, ta)
    return np.array(ta.astype("uint16"))

class ThreeDBackend:
    def __init__(self, width, height,
                 scale=600, fovx=None,
                 downSample=1, record=None):
        
        pipe = rec = mp.Queue(4)
        
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
        self.vertpoints = []
        self.vertnorms = []
        self.vertu = []
        self.vertv = []
        self.vtextures = []

        self.renderMask = None

        self.vertBones = []

        self.texAlphas = []
        self.vaNames = {}

        self.matShaders = {}
        
        self.skyTex = None
        self.skyHemiLight = [0.1,0.2,0.4]

        self.useOpSM = False

        self.recVideo = False

        self.buffer = np.zeros((self.H, self.W, 3), dtype="float")
        self.nFrames = 0
        self.tacc = False
        self.dofPos = np.array([0.,0,0])
        self.dsNum = -1000

        self.particleSystems = []
        self.psClouds = []
        
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
        self.ambLight = 0.2

        self.drawAxes = True
        self.axPoints = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                                 dtype="float")*0.25
        self.baseAxPoints = np.array(self.axPoints)
        
        self.timfps = np.zeros(12)
        self.numfps = 0
        self.maxFPS = 60

        self.selecting = False
        self.genNewBones = False

        if not os.path.isdir("Screenshots"):
            os.mkdir("Screenshots")

        if record == "rec":
            self.recvEvts = []
        elif record == "play":
            with open("evt.txt") as f:
                self.recvEvts = json.loads(f.read())
            self.evtNum = 0
        self.record = record

    def start(self):
        self.createObjects()

        self.vertPoints = [np.array(i) for i in self.vertpoints]
        self.vertNorms = [np.array(i) for i in self.vertnorms]
        self.vertU = [np.array(i) for i in self.vertu]
        self.vertV = [np.array(i) for i in self.vertv]
        self.vertLight = [np.ones((i.shape[0], 3)) for i in self.vertPoints]
        
        del self.vertpoints, self.vertnorms, self.vertu, self.vertv

        maxuv = max([i.shape[0] for i in self.vertU])
        Luv = len(self.vertU)
        pmax = max([ps.N for ps in self.particleSystems]) if len(self.particleSystems) > 0 else 1

        import Ops
        
        self.draw = Ops.CLDraw(self.skyTex.shape[1],
                               Luv, self.W, self.H, pmax)

        self.draw.setScaleCull(self.scale, self.cullAngleX, self.cullAngleY)

        self.skyTex = np.array(self.skyTex)
        self.draw.setSkyTex(self.skyTex[:,:,0],
                            self.skyTex[:,:,1],
                            self.skyTex[:,:,2],
                            self.skyTex.shape[1])
            
        for i in range(len(self.vtextures)):
            tex = self.vtextures[i]
            if "mip" in self.matShaders[i]:
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

        for ps in self.particleSystems:
            if ps.tex is not None:
                self.draw.setPSTex(getTexture(ps.tex), ps.tex)

        d = self.directionalLights[0]
        self.draw.setPrimaryLight(np.array([d["i"]]), np.array([viewVec(*d["dir"])]))
        
        bargs = (self.recP, self.evtQ, self.infQ,
                 self.W, self.H, self.mouseSensitivity,
                 self.downSample, self.recVideo)
        self.frontend = mp.Process(target=VS.runGUI, args=bargs)

        self.frontend.start()

        self.customizeFrontend()

    def updateRig(self, rig, ct, name, offset=0):
        bt = rig.b0.getTransform()
        self.draw.setBoneTransform(name, bt)
        for i in ct:
            if len(self.vertBones[i]) > 0:
                self.draw.boneTransform(0, self.vertPoints[i].shape[0]*3, i, name,
                                        offset)

        self.pendingShader = True

    def enableDOF(self, dofR=24, rad=0.2, di=4):
        self.bindKey("<F5>", self.dofScreenshot)
        self.dofRad = rad
        ps = []
        for y in range(-dofR, dofR, di):
            for x in range(int(-sqrt(dofR**2 - y**2)), int(sqrt(dofR**2 - y**2)), di):
                ps.append((x,y))
        self.dofPos = np.array(ps) * self.dofRad / dofR
        print("DOF samples:", len(ps))
        
    def dofScreenshot(self):
        self.ipos = self.pos
        self.dofTarget = self.pos + self.vv * self.zb[self.H//2, self.W//2]
        self.dsNum = self.frameNum
        self.tacc = True

    def lookAt(self, target):
        d = target - self.pos
        self.α = pi/2-atan2(*(d[2::-2]))
        self.β = -atan2(d[1], sqrt(d[0]**2 + d[2]**2))

    def stepParticles(self):
        for ps in self.particleSystems: ps.step()
    def resetParticles(self):
        for ps in self.particleSystems: ps.reset()

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
    def addParticleSystem(self, ps, isCloud=False):
        ps.setup()
        self.particleSystems.append(ps)
        self.psClouds.append(isCloud)
    
    def addSkyTex(self, n, t):
        self.skTexn = t
        
    def render(self):
        self.renderProfile(1)
        
        self.renderProfile(2)
        
        self.vc = self.viewCoords()
        self.vMat = np.stack((self.vv,self.vVhorz(),self.vVvert()))
        self.draw.setPos(self.vc)
        self.draw.setVM(self.vMat)

        self.renderProfile(3)
        
        self.draw.clearZBuffer()

        self.renderProfile(4)
        #if len(self.shadowCams) == 1:
        #    s = [0,0]
        #elif len(self.shadowCams) == 2:
        #    s = [0,1]
        #s = [0, "c"]
        s = [0,1]

##        shaders = {}
##        for i in range(len(self.vertPoints)):
##            if self.texUseAlpha[i]: shaders[i] += "alpha"
##            if self.texMip[i]: shaders[i] += "mip"
##            if self.texRefl[i]: shaders[i] += "refl"
            
        self.draw.drawAll(self.matShaders,
                          mask=self.renderMask,
                          shadowIds=s,
                          useOpacitySM=self.useOpSM)
        
        self.renderProfile(5)

        cc = []
        for i in range(len(self.particleSystems)):
            ps = self.particleSystems[i]
            pv = (ps.pos - self.pos) / np.linalg.norm((ps.pos - self.pos))
            if (pv @ self.vv) > (self.cullAngle - 0.2):
                cc.append((ps, self.psClouds[i]))

        cc = sorted(cc, key=lambda a: np.linalg.norm(a[0].pos - self.pos), reverse=True)

        for i in range(len(cc)):
            ps = cc[i][0]
            if cc[i][1]:
                self.draw.drawPSClouds("c", ps.pc, ps.opacity, ps.size, ps.tex,
                                       ps.pos, ps.randPos, self.skyHemiLight)
            elif ps.tex is not None:
                self.draw.drawPSTex(ps.pc, ps.color, ps.opacity, ps.size, ps.tex)
            else:
                self.draw.drawPS(ps.pc, ps.color, ps.opacity, ps.size)

        self.renderProfile(6)

        a = None
        if self.drawAxes:
            if (((self.axPoints - self.pos) @ self.vv) > 0).all():
                a = self.projectPoints(self.axPoints).astype("int")
        
        self.renderProfile(7)

        if self.tacc:
            x = self.draw.getFrame()
            x = np.stack(x[:3], axis=2)
            self.buffer += x
            self.nFrames += 1
        
        self.postProcess()
        
        self.renderProfile(8)
        
        result = self.draw.getFrame()
        rgb = np.stack(result[:3], axis=2)
        self.zb = result[3]
        
        #aa = Image.fromarray((rgb>>8).astype("uint8"), "RGB")
        #aa.save("test.png")
        
        self.renderProfile(9)
        
        return [rgb, a, self.selecting, (self.pos, self.vv)]

    def projectPoints(self, xyzn):
        dxy = (xyzn - self.pos) @ self.vMat.T

        x = dxy[:,1]
        y = dxy[:,2]
        s = self.scale
        d = s / dxy[:,0]
        dw, dh = self.W2, self.H2
        x2d = -x * d + dw
        y2d = y * d + dh
        #x2d = ne.evaluate("x * d * -s + dw")
        #y2d = ne.evaluate("y * d * s + dh")
        xy2d = np.stack((x2d, y2d), axis=1)
        return xy2d
        
    def simpleShaderVert(self, mask=None, updateLights=True):
        if mask is None:
            mask = [True] * len(self.vertU)
        dirI = np.array([d["i"] for d in self.directionalLights])
        dirD = np.array([viewVec(*d["dir"]) for d in self.directionalLights])
        if updateLights:
            if len(self.pointLights) == 0:
                pointI = 1
            else:
                pointI = np.array([p["i"] for p in self.pointLights])
            pointP = np.array([p["pos"] for p in self.pointLights])

            if len(self.spotLights) == 0:
                spotI = 1
            else:
                spotI = np.array([p["i"] for p in self.spotLights])
            spotD = np.array([p["vec"] for p in self.spotLights])
            spotP = np.array([p["pos"] for p in self.spotLights])
            
            self.draw.vertLight(mask, dirI, dirD, pointI, pointP,
                                spotI, spotD, spotP)
        else:
            self.draw.vertLight(mask, dirI, dirD)
           
    def shadowObjects(self):
        sobj = np.full((len(self.vertU),), False)
        for o in self.vertObjects:
            if o.castShadow:
                sobj[o.texNum] = True
        
        self.castObjs = sobj

    def setupShadowCams(self):
        for i in range(len(self.shadowCams)):
            s = self.shadowCams[i]
            g = "gi" in s
            self.draw.addShadowMap(i, s["size"], s["scale"],
                                   self.ambLight, g)

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

        #self.draw.shadowMap(i, castObjs, self.texUseAlpha, bias)
        
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
        x2d = x / d * -s + dw
        y2d = y / d * s + dh
        #x2d = ne.evaluate("x / d * -s + dw")
        #y2d = ne.evaluate("y / d * s + dh")

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

            ps = self.dofPos
            fn = self.frameNum - self.dsNum

            if fn < len(ps):
                self.pos = self.ipos + ps[fn][0]*self.vVhorz() + ps[fn][1]*self.vVvert()
                self.lookAt(self.dofTarget)
            elif fn == len(ps):
                self.tacc = False
                ts = time.strftime("%Y %b %d %H-%M-%S", time.gmtime())
                sn = "Screenshots/DOF_Screenshot " + ts + ".png"

                from Ops import cl, cq
                r = (self.buffer / self.nFrames).astype("uint16")
                cl.enqueue_copy(cq, self.draw.RO, np.array(r[:,:,0]))
                cl.enqueue_copy(cq, self.draw.GO, np.array(r[:,:,1]))
                cl.enqueue_copy(cq, self.draw.BO, np.array(r[:,:,2]))

                self.postProcess()
                r = np.stack(self.draw.getFrame()[:3], axis=2)

                i = PngImagePlugin.PngInfo()
                i.add_text("targ", " ".join([str(round(x, 3)) for x in self.dofTarget]))
                i.add_text("ipos", " ".join([str(round(x, 3)) for x in self.ipos]))

                x = np.clip(r, None, 255)
                Image.fromarray(x.astype("uint8")).save(sn, pnginfo=i)
                self.buffer = np.zeros((self.H, self.W, 3), dtype="float")
                self.nFrames = 0

            self.vv = self.viewVec()

            self.speed[:] = self.svert * self.vv
            self.speed += self.shorz * -self.vVhorz()
            self.pos += self.camSpeed * self.speed
            if (self.speed != 0).any() or self.panning:
                self.panning = False

            self.frameUpdate()
            self.renderProfile(0)
            
            #if self.pendingShader:
            #    self.simpleShaderVert()
            #    self.pendingShader = False
            
            r = self.render()
            data = ("render", np.array(r, dtype="object"))
            try:
                self.P.put_nowait(data)
            except Full:
                self.full += 1
            self.renderProfile(10)

            if self.record == "rec":
                self.recvEvts.append(("F", self.frameNum))

            if self.record != "play":
                while not self.evtQ.empty():
                    self.processEvent()
            else:
                a = True
                while a:
                    if self.recvEvts[self.evtNum] is None:
                        self.doQuit = True
                        a = False
                    elif self.recvEvts[self.evtNum][0] == "F":
                        if self.recvEvts[self.evtNum][1] > self.frameNum:
                            a = False
                    else:
                        self.processEvent(self.recvEvts[self.evtNum])
                    self.evtNum += 1
            
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

    def processEvent(self, e=None):
        if e is None:
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
        else:
            action = e
            if action[0] == "event":
                self.doEvent(action)
            elif action[0] == "eventk":
                self.moveKey(action[1])
            elif action[0] == "eventp":
                self.pan(action[1])
            elif action[0] == "select":
                self.cSelect(*action[1])
            elif action in self.handles:
                self.handles[action]()

        if self.record == "rec":
            self.recvEvts.append(action)

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
        if self.record == "rec":
            with open("evt.txt", "w") as f:
                f.write(json.dumps(self.recvEvts))

    def changeTitle(self, t):
        self.P.put(("title", str(t)))
    def bindKey(self, k, f):
        self.P.put(("key", k))
        self.handles[k] = f

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
