# 3D render
# Multiprocess
# Using PyOpenCL
# Multiple control
# Follow me

from tkinter import *
from math import sin, cos, sqrt, pi
import numpy
import numpy.random as nr
import random
import time

nr.seed(1)
random.seed(1)

from Compute import *
import multiprocessing as mp
import os
import json

from Rig import Rig

from AI import follow

class ThreeDApp(ThreeDBackend):
    def __init__(self):
        super().__init__(width=960, height=540, fovx=70,
                         downSample=1)
        
        self.changeTitle("AXI OpenCL 3D Visualizer")
        
        self.recVideo = False

        if self.recVideo: print("Recording!")

        self.α = 4.1
        self.β = -0.1
        
        self.camDist = 0.1
        self.pos = numpy.array([37.8,  4.4, 33.3])
        self.camSpeed = 0.2

        self.maxFPS = 33

        self.mouseSensitivity = 10

        self.ambLight = 0.08

        self.st = 0
        self.tt = 200
        self.drawAxes = False

        self.posen = 0
        self.poset = 0
        self.pstep = 0.4

        self.moving = False
        self.movingOld = False
        self.cr = 0
        self.cv = 0

        self.posen2 = 0
        self.poset2 = 0
        self.pstep2 = 0.4

        self.moving2 = False
        self.movingOld2 = False
        self.cr2 = 0
        self.cv2 = 0
        
        self.selchar = False

    def customizeFrontend(self):
        self.bindKey("r", "rR"); self.makeHandler("rR", self.rotateLight)
        self.bindKey("f", "ff"); self.makeHandler("ff", self.tgControl)

    def tgControl(self):
        self.selchar = not self.selchar
        
    def moveKey(self, key):
        if self.selchar:
            if key == "u":   self.moving = True
            elif key == "r": self.cv = -0.2
            elif key == "l": self.cv = 0.2
            elif key == "ZV": self.moving = False
            elif key == "ZH": self.cv = 0
        else:
            if key == "u":   self.moving2 = True
            elif key == "r": self.cv2 = -0.2
            elif key == "l": self.cv2 = 0.2
            elif key == "ZV": self.moving2 = False
            elif key == "ZH": self.cv2 = 0

    def stepPose(self):
        self.poset += self.pstep
        if (self.poset > 1):
            if self.posen == (len(self.poses) - 2):
                self.pstep = -self.pstep
            else:
                self.poset -= 1
                self.posen += 1
        elif (self.poset < 0):
            if self.posen == 0:
                self.pstep = -self.pstep
            else:
                self.poset += 1
                self.posen -= 1
                
        self.rig1.interpPose(self.poses[self.posen], self.poses[self.posen+1],
                            self.poset)
        self.updateRig(self.rig1, self.ctexn, "1")

    def stepPose2(self):
        self.poset2 += self.pstep2
        if (self.poset2 > 1):
            if self.posen2 == (len(self.poses) - 2):
                self.pstep2 = -self.pstep2
            else:
                self.poset2 -= 1
                self.posen2 += 1
        elif (self.poset2 < 0):
            if self.posen2 == 0:
                self.pstep2 = -self.pstep2
            else:
                self.poset2 += 1
                self.posen2 -= 1
                
        self.rig2.interpPose(self.poses[self.posen2], self.poses[self.posen2+1],
                            self.poset2)
        self.updateRig(self.rig2, self.ctexm, "2", offset=22)
        
    def rotateLight(self):
        #self.directionalLights[0]["dir"][0] += 0.01
        a = self.directionalLights[0]["dir"][1]
        a += 0.05
        self.directionalLights[0]["dir"][1] = (a % pi)
        self.directionalLights[1]["dir"][1] = (a % pi) + pi
        ti = abs(self.directionalLights[0]["dir"][1])

        #self.ambLight = 0.1 * sin(ti) + 0.05
        sc = self.shadowCams[0]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -40 * viewVec(*sc["dir"]) + numpy.array([40, 5, 40])
        self.updateShadowCam(0)
        sc["bias"] = 0.12 * abs(cos(ti))**2 + 0.03
        self.shadowMap(0, bias=sc["bias"])
        self.simpleShaderVert()

        d = self.directionalLights[0]
        self.draw.setPrimaryLight(np.array([d["i"]]), np.array([viewVec(*d["dir"])]))


    def createObjects(self):
        st = time.time()
        print("creating objects")
        
        mpath = "../Models/"
        
##        self.addVertObject(VertModel, [0,0,0],
##                           filename="Test3C.obj",
##                           animated=True,
##                           scale=1, shadow="C")
##        self.testv = self.vertObjects[-1]

        # "L2/L2.obj", # scale 0.75 offset 1.16
        # "LinkTP/Li.obj", # scale 0.75 offset 1.5
        # "Zelda2/Test5b.obj", # scale 0.33 offset 1.16 texMul 2
        # "Zelda/Ztest3.obj", # scale 1 offset 1.36 texMul 2

        self.addVertObject(VertModel, [0,0,0],
                           #filename=mpath+"L2/L2.obj", # scale 0.75 offset 1.16
                           filename=mpath+"LinkTP/Li.obj", # scale 0.75 offset 1.5
                           animated=True, mip=1, texMul=2,
                           scale=0.75, shadow="R")
        self.testv = self.vertObjects[-1]

        self.addVertObject(VertModel, [0,0,0],
                           filename=mpath+"Zelda/Ztest3.obj", # scale 1 offset 1.36
                           #filename=mpath+"Zelda2/Test5b.obj", # scale 0.33 offset 1.16
                           animated=True, boneOffset=22, texMul=2.5,
                           scale=1, shadow="R")
        self.testu = self.vertObjects[-1]
        
        self.addVertObject(VertTerrain, [-10, 0, -10],
                       heights="../Assets/Landscape3.png",
                       texture="../Assets/Grass.png", scale=(0.375, 5, 0.375),
                       shadow="CR", mip=1, uvspread=8)
        self.terrain = self.vertObjects[-1]

        f = "../AXI Creator/"
        self.addVertObject(VertModel, [30,5,25],
                           filename=f+"Well1a.obj", rot=(0,0,-pi/2),
                           scale=0.01, shadow="CR", reflect="a")

##        options = {"filename":mpath+"Rock/Rock.obj", "static":True,
##                   "texMode":None, "shadow":"CR", "mip":1,
##                   "maxWedgeDims":10000}
##        for i in range(0, 60, 4):
##            for j in range(0, 60, 4):
##                c = numpy.array((i, 0, j), dtype="float")
##                c += nr.rand(3) * 8
##                c[1] = self.terrain.getHeight(c[0],c[2])
##                r = random.random() * 6
##                r2 = random.random() * 6
##                r3 = random.random() * 6
##                s = random.random() * 0.4 + 0.1
##                self.addVertObject(VertModel, c, **options,
##                                scale=s, rot=(r3,r,r2))
##        
##        options = {"filename":"../Models/Tree4G/Tree_G.obj", "static":True,
##                   "texMode":None, "scale":0.001, "shadow":"CR",
##                   "maxWedgeDims":10000, "subDiv":1}
##        for i in range(0, 60, 10):
##            for j in range(0, 60, 10):
##                c = numpy.array((i, 0, j), dtype="float")
##                c += nr.rand(3) * 8
##                c[1] = self.terrain.getHeight(c[0],c[2])
##                r = random.random() * 6
##                self.addVertObject(VertModel, c, **options, rot=(0,r,0))
        
        self.directionalLights.append({"dir":[pi*2/3, 2.1], "i":[1.2,0.8,0.4]})
        self.directionalLights.append({"dir":[pi*2/3, 2.1+pi], "i":[0.2,0.3,0.1]})
        self.directionalLights.append({"dir":[0, pi/2], "i":[0.1,0.2,0.4]})

        self.shadowCams.append({"pos":[40, 5, 40], "dir":[pi/2, 1.1],
                                "size":4096, "scale":64})
        self.shadowCams.append({"pos":[40, 5, 40], "dir":[pi/2, 1.1],
                                "size":4096, "scale":400})
        
        self.makeObjects(0)
        
        self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Skybox1.png")
        self.skyBox.created()

        print("done in", time.time() - st, "s")
        
    def postProcess(self):
        #self.draw.blur()
        self.draw.gamma()
        
    def onMove(self):
        pass

    def shadowObjects2(self):
        sobj = np.full((len(self.vertU),), False)
        for o in self.ctexn: sobj[o] = True
        for o in self.ctexm: sobj[o] = True
        return sobj
        
    def onStart(self):

        self.cubeMap = CubeMap("../Skyboxes/Skybox1a.png", 2, False)
        a = self.cubeMap.texture.reshape((-1, 3))
        self.draw.setReflTex("a", a[:,0], a[:,1], a[:,2], self.cubeMap.m)
        self.draw.setHostSkyTex(self.cubeMap.rawtexture)

        ctexn = []
        c = self.testv
        ctexn.append(c.texNum)
        while c.nextMtl is not None:
            c = c.nextMtl
            ctexn.append(c.texNum)
        self.ctexn = ctexn

        ctexm = []
        c = self.testu
        ctexm.append(c.texNum)
        while c.nextMtl is not None:
            c = c.nextMtl
            ctexm.append(c.texNum)
        self.ctexm = ctexm
        
        #r = json.load(open("../Models/L2/L2.rig"))
        r = json.load(open("../Models/LinkTP/Li.rig"))
        #r = json.load(open("../Models/Zelda2/Test5b.rig"))
        self.rig1 = Rig(r, scale=0.75)

        r = json.load(open("../Models/Zelda/Ztest3.rig"))
        #r = json.load(open("../Models/Zelda2/Test5b.rig"))
        self.rig2 = Rig(r, scale=1)
        
        self.b1 = self.rig1.b0
        self.b2 = self.rig2.b0
        self.allBones1 = self.rig1.allBones
        self.allBones2 = self.rig2.allBones

        p = "../PyOpenCL23_B/"
        walks = ["Walk1.txt", "Walk2.txt", "Walk3.txt", "Walk4.txt"]
        self.poses = [json.load(open(p+f)) for f in walks]
        self.idle = json.load(open(p+"Idle.txt"))

        self.draw.initBoneTransforms("1", len(self.rig1.allBones))
        self.draw.initBoneTransforms("2", len(self.rig2.allBones))

        for b in range(len(self.allBones1)):
            for i in self.ctexn:
                self.draw.initBoneOrigin(self.rig1.allBones[b].origin, b, i)

        for b in range(len(self.allBones2)):
            for i in self.ctexm:
                self.draw.initBoneOrigin(self.rig2.allBones[b].origin, 22+b, i)

        self.b1.offset = np.array((30, 3, 30, 1.))
        self.updateRig(self.rig1, self.ctexn, "1")
        self.b1.offset[1] = self.terrain.getHeight(self.b1.offset[0],
                                                   self.b1.offset[2]) + 1.2

        self.b2.offset = np.array((33, 3, 29, 1.))
        self.updateRig(self.rig2, self.ctexm, "2", offset=22)
        
        self.shadowObjects()
        self.setupShadowCams()
        self.rotateLight()
        self.simpleShaderVert()

    def shadowChar(self):
        sc = self.shadowCams[1]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -20 * viewVec(*sc["dir"]) + numpy.array(self.b1.offset)[:3]
        self.updateShadowCam(1)
        self.shadowMap(1, self.shadowObjects2(), 0.02)
        
    def frameUpdate(self):
        if self.selchar: # controlling 1
            a = follow(self.b1.offset, self.b2.offset, 4, 1, self.moving2)
            self.cr2 = a[1]
            if a[0] is 0:
                self.moving2 = False
                self.rig2.importPose(self.idle, updateRoot=False)
                self.updateRig(self.rig2, self.ctexm, "2", offset=22)
            else: self.moving2 = True
        else:
            a = follow(self.b2.offset, self.b1.offset, 4, 1, self.moving)
            self.cr = a[1]
            if a[0] is 0:
                self.moving = False
                self.rig1.importPose(self.idle, updateRoot=False)
                self.updateRig(self.rig1, self.ctexn, "1")
            else: self.moving = True

        if self.moving:
            self.b1.offset[0] += 0.1*cos(self.cr)
            self.b1.offset[2] += 0.1*sin(self.cr)
            self.b1.offset[1] = self.terrain.getHeight(self.b1.offset[0],
                                                       self.b1.offset[2]) + 1.5
            self.b1.updateTM()
            self.stepPose()
        elif self.movingOld or self.cv != 0:
            self.rig1.importPose(self.idle, updateRoot=False)
            self.updateRig(self.rig1, self.ctexn, "1")
        
        self.cr += self.cv
        self.b1.rotate([0,self.cr,0])
        self.movingOld = self.moving

        if self.moving2:
            self.b2.offset[0] += 0.1*cos(self.cr2)
            self.b2.offset[2] += 0.1*sin(self.cr2)
            self.b2.offset[1] = self.terrain.getHeight(self.b2.offset[0],
                                                       self.b2.offset[2]) + 1.36
            self.b2.updateTM()
            self.stepPose2()
        elif self.movingOld2 or self.cv2 != 0:
            self.rig2.importPose(self.idle, updateRoot=False)
            self.updateRig(self.rig2, self.ctexm, "2", offset=22)

        self.cr2 += self.cv2
        self.b2.rotate([0,self.cr2,0])
        self.movingOld2 = self.moving2

        self.shadowChar()
        self.simpleShaderVert(None, False)

if __name__ == "__main__":
    app = ThreeDApp()
    try:
        print("starting")
        app.start()
        print("running")
        app.runBackend()
        app.finish()
        print("finished")
        print("avg spf:", app.totTime / app.frameNum)
    except:
        raise
    finally:
        with open("Profile.txt", "w") as f:
            f.write("Backend profile\n\n")
            f.write(app.printProfile())
