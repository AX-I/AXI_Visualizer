# 3D render
# Multiprocess
# Using PyOpenCL
# Instruments

from tkinter import *
from math import sin, cos, sqrt, pi
import numpy
import numpy.random as nr
import random
import time

# 1, 6
nr.seed(6)
random.seed(6)

from Compute import *
import multiprocessing as mp
import os
import json

from Rig import Rig

class ThreeDApp(ThreeDBackend):
    def __init__(self):
        super().__init__(width=1280, height=720, fovx=72,
                         downSample=1)
        # Resolutions
        # 7680 x 5760  (4:3) (fov 65)
        # 3840 x 2160  (16:9)
        # 2880 x 2160  (4:3)
        # 2400 x 1800  (4:3)
        # 1920 x 1080  (16:9) (fs)
        # 1280 x 720   (16:9) (fs)
        # 1200 x 900   (4:3)
        # 960 x 540    (16:9)
        # 640 x 360    (16:9)
        
        self.changeTitle("AXI OpenCL 3D Visualizer")

        #self.α = 4.25; self.β = 0.08
        #self.pos = numpy.array([43.02, 5.18, 36.45])
        self.α = 4.24; self.β = 0.08
        #self.pos = numpy.array([42.5, 5.13, 35.72])
        self.pos = numpy.array([41.44, 5.11, 34.73])
        self.camSpeed = 0.1

        self.mouseSensitivity = 10

        self.ambLight = 0.001

        self.maxFPS = 24

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

    def customizeFrontend(self):
        self.bindKey("r", "rR"); self.makeHandler("rR", self.rotateLight)
        self.bindKey("c", "cC"); self.makeHandler("cC", self.testcubemap)
        self.bindKey("g", "gG"); self.makeHandler("gG", self.doGI)

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
        
    def rotateLight(self):
        a = self.directionalLights[0]["dir"][1]
        a += 0.05
        self.directionalLights[0]["dir"][1] = (a % pi)
        self.directionalLights[1]["dir"][1] = (a % pi) + pi
        ti = abs(self.directionalLights[0]["dir"][1])

        sc = self.shadowCams[0]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -40 * viewVec(*sc["dir"]) + numpy.array([33, 5, 30])
        self.updateShadowCam(0)
        self.shadowMap(0, bias=0.02)
        sc = self.shadowCams[1]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -40 * viewVec(*sc["dir"]) + numpy.array([33, 5, 30])
        self.updateShadowCam(1)
        self.shadowMap(1, bias=0.08)
        
        self.simpleShaderVert()
        
        d = self.directionalLights[0]
        self.draw.setPrimaryLight(np.array([d["i"]]), np.array([viewVec(*d["dir"])]))
        
    def doGI(self):
        self.draw.clearShadowMap(0)
        self.draw.drawDirectional(0, self.castObjs)

        self.g = self.draw.getGIM(0)
        sc = self.shadowCams[0]
        zb = self.g[1]
        nb = self.g[2]
        vm = viewMat(*sc["dir"])
        self.spotLights = []
        for i in range(0, sc["size"], 20):
            for j in range(0, sc["size"], 20):
                cz = zb[j, i] - 0.08
                if cz < 6000:
                    p = np.array(sc["pos"], dtype="float") + cz*vm[0]
                    p += (i-sc["size"]/2)/sc["scale"] * vm[1]
                    p += -(j-sc["size"]/2)/sc["scale"] * vm[2]
                    d = nb[j, i][:3]
                    p += 0.01*d
                    ix = self.g[0][j, i] / (2**8) / 255
                    self.spotLights.append({"i":0.012 * ix, "pos":p, "vec":-d})
        #print(len(self.spotLights))
        #self.simpleShaderVert()
        #Image.fromarray((self.g[0]>>8).astype("uint8")).save("test.png")

    def createObjects(self):
        st = time.time()
        print("creating objects")
        
        mpath = "../Models/"
        
        self.addVertObject(VertSphere, [34, 6, 25.8], n=32, scale=1.6,
                           texture="../Assets/Earth2a.png",
                           reflect="Earth",
                           rot=(0, 2*pi/3, 0), shadow="CR")
        self.reflSphere = self.vertObjects[-1]
        
        self.addVertObject(VertTerrain, [-10, 0, -10],
                       heights="../Assets/Landscape3.png",
                       texture="../Assets/Grass1.png", scale=(0.375, 5, 0.375),
                       shadow="CR", mip=1, uvspread=4)
        self.terrain = self.vertObjects[-1]
        
        
        options = {"filename":mpath+"Rock/Rock.obj", "static":True,
                   "texMode":None, "shadow":"CR", "mip":1,
                   "maxWedgeDims":10000, "subDiv":1}
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
        
        treef = "../TreeModels/"
        options = {"filename":treef+"Tree4G/Tree_G.obj", "static":True,
                   "texMode":None, "scale":0.002, "shadow":"CR",
                   "maxWedgeDims":10000, "subDiv":1}
        for i in range(0, 60, 12):
            for j in range(0, 60, 12):
                c = numpy.array((i, 0, j), dtype="float")
                c += nr.rand(3) * 8
                c[1] = self.terrain.getHeight(c[0],c[2]) - 0.2
                r = random.random() * 3
                self.addVertObject(VertModel, c, **options, rot=(0,r,0))
        
##        grt = "../Assets/Grass2.png"
##        gra = "../Assets/Grass2A.png"
##        options = {"n":4, "texture":grt, "alpha":gra, "norm":[0,1,0],
##                   "h1":[0,0.6,0], "h2":[0,0,1], "origin":[0,0,0.5],
##                   "static":True, "shadow":"R", "texMode":"safe"}
##        for i in range(0, 60, 1):
##            for j in range(0, 60, 1):
##                c = numpy.array((i, 0, j), dtype="float")
##                c += nr.rand(3) * 2 - 1
##                c[1] = self.terrain.getHeight(c[0],c[2]) - 0.05
##                r = random.random() * 2
##                r2 = random.random() * 0.2
##                self.addVertObject(VertPlane, c, **options, rot=(r2, r, r2))
##                self.addVertObject(VertPlane, c, **options, rot=(r2, 2/3*pi+r, r2))
##                self.addVertObject(VertPlane, c, **options, rot=(r2, 4/3*pi+r, r2))

        blank = "../Assets/Blank.png"
        blank1 = "../Assets/Blank1.png"
        red = "../Assets/Col/Red.png"
        blue = "../Assets/Col/Blue.png"
        self.addVertObject(VertSphere, [30, 2, 33.7], n=48,
                           scale=2, texture=blank, texMul=1.6, shadow="C", mip=1)
        self.addVertObject(VertSphere, [15, 3, 47], n=48,
                           scale=2, texture=blank, shadow="C")
        self.addVertObject(VertSphere, [34, 4, 35.7], n=32,
                           scale=1.3, texture=blank1, texMul=1.2, shadow="C", mip=1)
        self.addVertObject(VertSphere, [23, 2, 23], n=48,
                           scale=1.5, texture=blank, shadow="C")
        
        self.addVertObject(VertPlane, [30, 1, 27], n=24,
                           h2=[0,0,2], h1=[0,4,0], texture=blank, shadow="C")
        self.addVertObject(VertPlane, [29.9, 1, 27], n=24,
                           h1=[0,0,2], h2=[0,4,0], texture=blank, shadow="C")
        
        self.addVertObject(VertPlane, [31.5, 1, 24], n=24,
                           h2=[-1.5,0,3], h1=[0,4,0], texture=red,
                           texMul=15.9, shadow="C", mip=1)
        self.addVertObject(VertPlane, [31.4, 1, 24], n=24,
                           h1=[-1.5,0,3], h2=[0,4,0], texture=red,
                           texMul=15.9, shadow="C")
        self.addVertObject(VertPlane, [30, 1, 29], n=24,
                           h2=[1,0,3], h1=[0,3,0], texture=blue,
                           texMul=3.2, shadow="C", mip=1)
        self.addVertObject(VertPlane, [29.9, 1, 29], n=24,
                           h1=[1,0,3], h2=[0,3,0], texture=blue,
                           shadow="C")

        self.addVertObject(VertPlane, [17, 3, 33], n=24,
                           h2=[0,0,3], h1=[1,3,0], texture=blank,
                           shadow="C", mip=1)
        self.addVertObject(VertPlane, [16.9, 3, 33], n=24,
                           h1=[0,0,3], h2=[1,3,0], texture=blank,
                           shadow="C", mip=1)

##        options = {"filename":mpath+"Trees/Fir3/firtree3.obj", "static":True,
##                   "texMode":None, "shadow":"CR",
##                   "maxWedgeDims":10000}
##        for i in range(-8, 5, 6):
##            for j in range(0, 50, 9):
##                c = numpy.array((i, 0, j), dtype="float")
##                c += nr.rand(3) * 4
##                c[1] = self.terrain.getHeight(c[0],c[2]) - 0.2
##                r = random.random() * 8
##                s = 1 + random.random() * 1
##                self.addVertObject(VertModel, c, **options, rot=(-pi/2,r,0), scale=s)
        
        self.addVertObject(VertModel, [35.5,2.44,31.6],
                           filename=mpath+"LinkTrumpet_1.obj",
                           scale=1, shadow="CR")
        self.addVertObject(VertModel, [35,2.3,33.9],
                           filename=mpath+"ZeldaViolin_a.obj",
                           scale=1.2, shadow="CR", rot=(0,-pi/2-0.05,0), subDiv=1)
        self.addVertObject(VertModel, [35.5,2.83,29.6],
                           filename=mpath+"LinkViolin_1.obj",
                           scale=1, shadow="CR", rot=(-0.1,-pi/3,0), subDiv=1)
        self.addVertObject(VertModel, [33,3.33,32.1],
                           filename=mpath+"Stormtrooper/StormtrooperViolin_1.obj",
                           scale=1, shadow="CR", rot=(0,pi/6,0))

        self.pointLights.append({"i":[0.2,0,0], "pos":[10,5,10]})
        self.spotLights.append({"i":[0,0,0.5], "pos":[20,5,10], "vec":[0,-1,0]})

        self.directionalLights.append({"dir":[pi*2/5, 2.1], "i":[1.8,1.4,0.7]})
        self.directionalLights.append({"dir":[pi*2/5, 2.1+pi], "i":[0.0,0.0,0.0]})
        self.directionalLights.append({"dir":[0, pi/2], "i":[0.1,0.2,0.5]})


        self.shadowCams.append({"pos":[40, 5, 40], "dir":[pi/2, 1.1],
                                "size":6144, "scale":384, "gi":1})
        self.shadowCams.append({"pos":[40, 5, 40], "dir":[pi/2, 1.1],
                                "size":6144, "scale":96})
        
        self.makeObjects(0)
        
        self.skyBox = TexSkyBox(self, self, 12, "../Skyboxes/Skybox1.png",
                                rot=(0,pi/2,0))
        self.skyBox.created()

        print("done in", time.time() - st, "s")
        
    def postProcess(self):
        self.draw.blur()
        self.draw.gamma()
        
    def onMove(self):
        pass
        
    def onStart(self):
        self.cubeMap = CubeMap("../Skyboxes/Skybox1a.png", 2, False)
        a = self.cubeMap.texture.reshape((-1, 3))
        self.draw.setReflTex("Earth", a[:,0], a[:,1], a[:,2], self.cubeMap.m)
        
        self.draw.setHostSkyTex(self.cubeMap.rawtexture)
        
        self.shadowObjects()
        self.setupShadowCams()
        self.rotateLight()

    def frameUpdate(self):
        if self.frameNum == 0:
            self.doGI()
            print("g")
            #self.rotateLight()
            #print("r")
            self.shadowMap(0, bias=0.02)
            self.testcubemap()
            print("c")
    
if __name__ == "__main__":
    mp.set_start_method("forkserver")
    app = ThreeDApp()
    app.start()
    app.runBackend()
    app.finish()
    print("avg spf:", app.totTime / app.frameNum)
