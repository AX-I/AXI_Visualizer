# 3D render
# Multiprocess
# Using PyOpenCL
# Procedural sky
# Cloud clumps

from tkinter import *
from math import sin, cos, sqrt, pi
import numpy
import numpy.random as nr
import random
import time

nr.seed(3)
random.seed(3)

from Compute import *
import multiprocessing as mp
import os
import json
from ParticleSystem import ParticleSystem, SpiralParticleSystem, CentripetalParticleSystem, ContinuousParticleSystem

from Rig import Rig

class ThreeDApp(ThreeDBackend):
    def __init__(self):
        super().__init__(width=1024, height=640, fovx=80,
                         downSample=1)
        
        self.changeTitle("AXI OpenCL 3D Visualizer")

        self.α = 6.05
        self.β = -0.4
        
        self.camDist = 0.1
        #self.pos = numpy.array([32.0,  14.5, 36.3])
        self.pos = numpy.array([85.84, 9.63, 18.93])
        self.camSpeed = 0.5

        self.mouseSensitivity = 10

        self.ambLight = 0.001

        self.maxFPS = 24

        self.st = 0
        self.tt = 200
        self.drawAxes = False

        self.posen = 0
        self.poset = 0
        self.pstep = 0.4
        self.cspeed = 0.1

        self.moving = False
        self.movingOld = False
        self.cr = 0
        self.cv = 0

        self.charH = 0
        self.camLock = False

        self.sunCol = np.array(Image.open("../Assets/SunGradient.png"))[0,:,:3] / 128
        self.skyCol = np.array(Image.open("../Assets/EarthClearSky2.png"))[0,:,:3] / 128

    def customizeFrontend(self):
        self.bindKey("r", "rR"); self.makeHandler("rR", self.rotateLight)
        self.bindKey("R", "Rr"); self.makeHandler("Rr", self.unrotateLight)

        self.bindKey("f", "ff"); self.makeHandler("ff", self.camlk)
        
        self.bindKey("c", "cC"); self.makeHandler("cC", self.testcubemap)

        self.bindKey("v", "vv"); self.makeHandler("vv", self.resetParticles)
        
    def camlk(self):
        self.camLock = not self.camLock

##    def moveKey(self, key):
##        if key == "u":   self.moving = True
##        elif key == "r": self.cv = -0.2
##        elif key == "l": self.cv = 0.2
##        elif key == "ZV": self.moving = False
##        elif key == "ZH": self.cv = 0

    def stepPose(self):
        self.poset += self.pstep
        if self.poset >= (len(self.poses) - 1):
            self.pstep = -self.pstep
            self.poset += self.pstep
        elif self.poset < 0:
            self.pstep = -self.pstep
                
        self.rig.interpPose(self.poses[int(self.poset)],
                            self.poses[int(self.poset)+1],
                            self.poset-int(self.poset))
        self.charH = 1.2 - 0.02*abs(self.poset - 1.5)
        self.updateRig(self.rig, self.ctexn, "L")

    def testcubemap(self):
        S = 512
        mask = self.reflSphere.texNum
        c = self.reflSphere.coords
        cM = self.draw.drawCubeMap(512, c,
                                   self.texUseAlpha, self.texShadow,
                                   self.texMip, self.texRefl,
                                   maskNum=mask)
        #self.test = cM
        cM = CubeMap(cM, 2)
        a = cM.texture.reshape((-1, 3))
        self.draw.setReflTex("Earth", a[:,0], a[:,1], a[:,2], cM.m)

    def unrotateLight(self):
        sd = self.directionalLights[0]["dir"]
        oldPos = -30 * viewVec(*sd)
        self.directionalLights[0]["dir"][1] -= 0.1
        self.sunPoints += np.array([[-30 * viewVec(*sd) - oldPos,],])
        self.sunAvgPos += np.array([-30 * viewVec(*sd) - oldPos,]).T
        self.rotateLight()
        
    def rotateLight(self):
        sd = self.directionalLights[0]["dir"]
        oldPos = -30 * viewVec(*sd)
        
        ti = self.directionalLights[0]["dir"][1]
        ti += 0.02
        self.directionalLights[0]["dir"][1] = (ti % (2*pi))
        self.directionalLights[1]["dir"][1] = (ti % (2*pi)) + pi

        if (ti > pi*1.2) & (ti < pi*1.5):
            ti = pi*1.8

        #self.ambLight = 0.1 * sin(ti) + 0.05
        sc = self.shadowCams[0]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -40 * viewVec(*sc["dir"]) + numpy.array([40, 5, 40])
        self.updateShadowCam(0)
        sc["bias"] = 0.1 * abs(cos(ti))**2 + 0.1
        self.shadowMap(0, bias=sc["bias"])

        v = abs(ti / pi - 0.5)
        v = v if ti<(1.5*pi) else 2-v

        cSunCol = self.sunCol[int(v * len(self.sunCol))]
        d = self.directionalLights[0]
        d["i"] = cSunCol / 1.4
        self.draw.setPrimaryLight(np.array([d["i"]]), np.array([viewVec(*d["dir"])]))

        self.skyV[:] = 0.1 + 0.9*v

        d = self.directionalLights[1]
        d["i"] = cSunCol * np.array([0.1, 0.2, 0.04]) * 0.5
        d = self.directionalLights[2]
        d["i"] = self.skyCol[int(v * len(self.sunCol))] / 3
        self.skyHemiLight = d["i"]

        self.simpleShaderVert()

        sd = self.directionalLights[0]["dir"]
        d = -40 * viewVec(*sd) + numpy.array([40, 50, 40])
        self.draw.placeShadowMap("c", d, sd)
        self.draw.clearShadowMap("c")
        self.draw.clearShadowOpMap("c")
        for ps in self.particleSystems:
            self.draw.shadowClouds("c", ps.pc, ps.shSize, ps.tex,
                                          ps.pos, ps.randPos)
            self.draw.shadowCloudsOpacity("c", ps.pc, ps.shSize, ps.tex,
                                          ps.pos, ps.randPos)

        self.sunPoints += np.array([[-30 * viewVec(*sd) - oldPos,],])
        self.sunAvgPos += np.array([-30 * viewVec(*sd) - oldPos,]).T
        
    def createObjects(self):
        st = time.time()
        print("creating objects")
        
        mpath = "../Models/"
        
        self.addVertObject(VertModel, [0,0,0],
                           filename="../Models/L2/L2.obj",
                           animated=True,
                           scale=0.75, shadow="R")
        self.testv = self.vertObjects[-1]
        
##        self.addVertObject(VertSphere, [33, 6, 26], n=32, scale=0.2,
##                           texture="../Assets/Earth2a.png",
##                           #texture="../Assets/Col/Gray.png",
##                           reflect="Earth",
##                           rot=(0, 2*pi/3, 0), shadow="CR")
##        self.reflSphere = self.vertObjects[-1]
        
        self.addVertObject(VertSphere, [33, 6, 26], n=32, scale=2,
                           texture="../Assets/Blank1.png", shadow="C")
        
        self.addVertObject(VertTerrain, [-10, 0, -10],
                       heights="../Assets/Landscape3.png",
                       texture="../Assets/Grass.png", scale=(0.375, 6, 0.375),
                       shadow="CR", mip=1, uvspread=4)
        self.terrain = self.vertObjects[-1]

##        self.addVertObject(VertModel, [27, 1, 29],
##                           filename=treef+"EuLind/European_Linden_LowPoly.obj",
##                           scale=2, shadow="CR", subDiv=1)
##        self.addVertObject(VertModel, [23, 1.5, 29],
##                           filename=treef+"RedMap/Red_Maple.obj",
##                           scale=3, shadow="CR", subDiv=1)
##        self.addVertObject(VertModel, [21, 1.5, 27], rot=(-pi/2,0,0),
##                           filename=mpath+"Trees/Fir3/firtree3.obj",
##                           scale=1.5, shadow="CR")

        
        options = {"filename":mpath+"Rock/Rock.obj", "static":True,
                   "texMode":None, "shadow":"CR", "mip":1,
                   "maxWedgeDims":10000}
        for i in range(0, 60, 6):
            for j in range(0, 60, 6):
                c = numpy.array((i, 0, j), dtype="float")
                c += nr.rand(3) * 8
                c[1] = self.terrain.getHeight(c[0],c[2])
                r = random.random() * 6
                r2 = random.random() * 6
                r3 = random.random() * 6
                s = random.random() * 0.4 + 0.1
                self.addVertObject(VertModel, c, **options,
                                scale=s, rot=(r3,r,r2))
        
##        options = {"filename":mpath+"Tree4G/Tree_G.obj", "static":True,
##                   "texMode":None, "shadow":"CR",
##                   "maxWedgeDims":1000, "subDiv":1}
##        for i in range(0, 60, 8):
##            for j in range(0, 60, 8):
##                c = numpy.array((i, 0, j), dtype="float")
##                c += nr.rand(3) * 8
##                c[1] = self.terrain.getHeight(c[0],c[2]) - 0.1
##                r = random.random() * 6
##                s = 0.0008 + random.random() * 0.0003
##                self.addVertObject(VertModel, c, **options, scale=s, rot=(0,r,0))

##        for h in range(4, 1, -1):
##            for i in range(0, 30, 9):
##                for j in range(0, 30, 6):
##                    c = nr.rand(3) * np.array((5000, 20, 5000)) + \
##                        np.array((-2000,30+120*h,-2000))
##                    if (c[0] > -700) and (c[0] < 800) and \
##                       (c[2] > -700) and (c[2] < 800):
##                        c[0] += (0.5-nr.rand())*1000
##                        c[2] += (0.5-nr.rand())*1000
##                    r = nr.rand(1)
##                    n = int(3000 * (r+1)**2)
##                    op = 0.045 + 0.03*nr.rand(1)
##                    r = np.array((5,2,5)) * (r+1) * 8
##                    ps = ParticleSystem(c, [1.1, 0], opacity=op,
##                                        vel=0.7, randVel=0.0, drag=0,
##                                        randPos=r, randColor=0,
##                                        color=(10, 10, 10),
##                                        nParticles=n, lifespan=2000,
##                                        shSize=7,
##                                              tex="../Flares/L4.png", size=1000)
##                    self.addParticleSystem(ps, True)
        
        for h in range(3, 1, -1):
            for i in range(0, 30, 8):
                for j in range(0, 30, 8):
                    c = nr.rand(3) * np.array((850, 20, 850)) + \
                        np.array((-400,30+50*h,-400))
                    r = nr.rand(1)
                    n = int(6000 * (r+1)**2)
                    op = 0.045 + 0.03*nr.rand(1)
                    r = np.array((2.5,1.6,2.5)) * (r+1) * 6
                    ps = ParticleSystem(c, [1.1, 0], opacity=op,
                                        randDir=0.8,
                                        vel=0.7, randVel=0.004, drag=0,
                                        randPos=r, randColor=0,
                                        color=(100, 100, 100),
                                        nParticles=n, lifespan=2000,
                                        shSize=2,
                                              tex="../Flares/L4.png", size=500)
                    self.addParticleSystem(ps, True)
        
        
##        ps = CentripetalParticleSystem([16, 5, 20], [1.1-pi/2, 0.1-pi/2],
##                                       opacity=0.1,
##                            vel=0., randVel=0.02, randPos=0.05,
##                                       tex="../Flares/L4.png",
##                            color=(200, 0, 0), colorOverLife=(250,240,100),
##                            randColor=20, nParticles=6000, lifespan=100,
##                            size=4, f=0.01, r=1.2, cc=1)
##        self.addParticleSystem(ps)
        
##        ps = SpiralParticleSystem([16, 5, 20], [1.1, 0.1], opacity=0.0002,
##                            vel=0.3, randPos=0.15, color=(255, 0, 255),
##                                  colorOverLife=(0,255,255),
##                            nParticles=4000, lifespan=400,
##                                  tex="../Assets/Logo2.png",
##                            f=0.02, r=0.5, cc=1, turns=4, offset=0, size=8)
##        self.addParticleSystem(ps)
##        ps = SpiralParticleSystem([16, 5, 20], [1.1, 0.1], opacity=0.0002,
##                            vel=0.3, randPos=0.15, color=(255, 0, 255),
##                                  colorOverLife=(0,255,100),
##                                  tex="../Flares/L4.png",
##                            nParticles=6000, lifespan=400,
##                            f=0.02, r=0.5, cc=1, turns=4, offset=0.33, size=5)
##        self.addParticleSystem(ps)
##        ps = SpiralParticleSystem([16, 5, 20], [1.1, 0.1], opacity=0.0002,
##                            vel=0.3, randPos=0.15, color=(255, 0, 255),
##                                  colorOverLife=(0,255,100),
##                                  tex="../Flares/L7.jpeg",
##                            nParticles=6000, lifespan=400,
##                            f=0.02, r=0.5, cc=1, turns=4, offset=0.66, size=5)
##        self.addParticleSystem(ps)
        
        options = {"n":6, "texture":"../Assets/Blank.png", "scale":0.05,
                   "invertNorms":True, "texMul":0.02}

        self.pointLights.append({"i":[1,0.7,0], "pos":[16,7,20]})
        
        self.directionalLights.append({"dir":[pi*2/3, 0.93], "i":[1.0,0.4,0.2]})
        self.directionalLights.append({"dir":[pi*2/3, 0.93+pi], "i":[0.3,0.2,0.1]})
        self.directionalLights.append({"dir":[0, pi/2], "i":[0,0.2,0.4]})

        self.spotLights.append({"i":[0,0,0.5], "pos":[20,5,10], "vec":[0,-1,0]})

        self.shadowCams.append({"pos":[80, 5, 80], "dir":[pi/2, 1.1],
                                "size":8192, "scale":64})
        self.shadowCams.append({"pos":[40, 5, 40], "dir":[pi/2, 1.1],
                                "size":1024, "scale":128})
        
        self.makeObjects(0)
        
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Skybox1.png",
        #                        )#rot=(0,pi/2,0))
        #self.skyBox = TexSkyBox(self, 12, "../Assets/Sepia.png")
        self.skyBox = TexSkyHemiSphere(self, 32, "../Assets/EarthClearSky2.png")
        self.skyBox.created()
        
        sd = self.directionalLights[0]["dir"]
        d = -30 * viewVec(*sd)
        self.sun = Sun(self, 16, "../Assets/Blank.png", (0,0,0), d, "")
        self.sun.created()

        print("done in", time.time() - st, "s")
        
    def postProcess(self):
        self.draw.blur()
        self.draw.gamma()
        
    def onMove(self):
        pass

    def stepParticles(self):
        for ps in self.particleSystems:
            ps.step()
    def resetParticles(self):
        for ps in self.particleSystems:
            ps.reset()

    def shadowObjects2(self):
        sobj = np.full((len(self.vertU),), False)
        for o in self.ctexn:
                sobj[o] = True
        return sobj
        
    def onStart(self):
        sd = self.directionalLights[0]["dir"]
        d = -40 * viewVec(*sd) + numpy.array([40, 50, 40])
        self.draw.addShadowMap("c", 1024, 1/2)
        self.draw.placeShadowMap("c", d, sd)
        self.draw.clearShadowMap("c")
        self.draw.addOpacityMap("c")
        self.opSM = True
        self.shadowIDs = [0, "c"]
        
        ctexn = []
        c = self.testv
        ctexn.append(c.texNum)
        while c.nextMtl is not None:
            c = c.nextMtl
            ctexn.append(c.texNum)
        self.ctexn = ctexn

        self.ctMask = np.array([False]*len(self.vertU))
        for i in ctexn: self.ctMask[i] = True
        
        self.cubeMap = CubeMap("../Skyboxes/Skybox1a.png", 2, False)
        a = self.cubeMap.texture.reshape((-1, 3))
        self.draw.setReflTex("Earth", a[:,0], a[:,1], a[:,2], self.cubeMap.m)
        
        self.draw.setHostSkyTex(self.cubeMap.rawtexture)
        
        r = json.load(open("../Models/L2/L2.rig"))
        self.rig = Rig(r, scale=0.75)
        self.b0 = self.rig.b0
        self.allBones = self.rig.allBones

        pp = "../PyOpenCL23_B/"
        walks = ["Walk1.txt", "Walk2.txt", "Walk3.txt", "Walk4.txt"]
        self.poses = [json.load(open(pp+f)) for f in walks]
        self.idle = json.load(open(pp+"Idle.txt"))

        self.draw.initBoneTransforms("L", len(self.rig.allBones))

        for b in range(len(self.allBones)):
            for i in range(len(self.vertBones)):
                vb = self.vertBones[i]
                if len(vb) > 0:
                    self.draw.initBoneOrigin(self.rig.allBones[b].origin, b, i)

        self.b0.offset = np.array((20, 5, 20, 1.))
        self.updateRig(self.rig, self.ctexn, "L")
        self.b0.offset[1] = self.terrain.getHeight(self.b0.offset[0],
                                                   self.b0.offset[2]) + 1.8
        self.shadowObjects()
        self.setupShadowCams()
        self.rotateLight()
        self.simpleShaderVert()

    def shadowChar(self):
        sc = self.shadowCams[1]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -20 * viewVec(*sc["dir"]) + numpy.array(self.b0.offset)[:3]
        self.updateShadowCam(1)
        self.shadowMap(1, self.shadowObjects2(), 0.05)
        
    def frameUpdate(self):
        if self.frameNum == 0:
            for ps in self.particleSystems:
                self.draw.shadowClouds("c", ps.pc, ps.shSize, ps.tex,
                                          ps.pos, ps.randPos)
                self.draw.shadowCloudsOpacity("c", ps.pc, ps.shSize, ps.tex,
                                              ps.pos, ps.randPos)
        
##        self.stepParticles()
##        self.rotateLight()

##        sd = self.directionalLights[0]["dir"]
##        vd = 0.1*(self.frameNum%100) * anglesToCoords([1.1, 0])
##        d = -40 * viewVec(*sd) + numpy.array([40, 50, 40]) + vd
##        self.draw.placeShadowMap("c", d, sd)
        
##        if self.frameNum % 20 == 0:
##            self.draw.clearShadowMap("c")
##            self.draw.clearShadowOpMap("c")
##            for ps in self.particleSystems:
##                self.draw.shadowClouds("c", ps.pc, ps.shSize, ps.tex)
##                self.draw.shadowCloudsOpacity("c", ps.pc, ps.shSize, ps.tex)
        
        if self.moving:
            self.b0.offset[0] += self.cspeed*cos(self.cr)
            self.b0.offset[2] += self.cspeed*sin(self.cr)
            self.b0.offset[1] = self.terrain.getHeight(self.b0.offset[0],
                                                       self.b0.offset[2]) + self.charH
            self.b0.updateTM()
            self.stepPose()
            self.simpleShaderVert(self.ctMask, False)
            self.shadowChar()
        elif self.movingOld:
            self.rig.importPose(self.idle, updateRoot=False)
            self.updateRig(self.rig, self.ctexn, "L")
            self.simpleShaderVert(self.ctMask, False)
            self.shadowChar()
        elif self.cv != 0:
            self.updateRig(self.rig, self.ctexn, "L")
            self.simpleShaderVert(self.ctMask, False)
            self.shadowChar()
        
        self.cr += self.cv
        self.b0.rotate([0,self.cr,0])

        if self.camLock:
            self.pos = self.b0.offset[:3] + -5*self.vv + np.array([0, 0.02*abs(self.poset - 1.5),0])

        self.movingOld = self.moving

    def cs(self):
        a = self.draw.getSHM("c")
        b = Image.fromarray(np.clip(a, 0, 255).astype("uint8"), "L")
        b.show()

if __name__ == "__main__":
    app = ThreeDApp()
    try:
        print("starting")
        app.start()
        print("running")
        app.runBackend()
        app.finish()
        print("finished")
        app.fps()
    except:
        raise
    finally:
        with open("Profile.txt", "w") as f:
            f.write("Backend profile\n\n")
            f.write(app.printProfile())
