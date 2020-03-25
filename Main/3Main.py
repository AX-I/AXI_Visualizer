# 3D render
# Multiprocess
# Using PyOpenCL
# Global Illumination
# Minecraft models

# Testing tiled scanline

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

class ThreeDApp(ThreeDBackend):
    def __init__(self):
        super().__init__(width=960, height=640, fovx=75,
                         downSample=1)
        
        self.changeTitle("AXI OpenCL 3D Visualizer")

        #self.pos = np.array([4.462, 3.811, 1.581])
        #self.lookAt([-985, -119, -125])

        self.α = 5.438; self.β = -0.56
        self.pos = np.array([2.46478414, 0.16076738, 0.60498475])
        
        #self.α = 7.22; self.β = 0.055
        #self.pos = np.array([-2.04,  0.64, -3.07])
        self.camSpeed = 0.2
        self.mouseSensitivity = 10

        self.ambLight = 0.001

        self.maxFPS = 24

        self.st = 0
        self.tt = 200
        self.drawAxes = False

    def customizeFrontend(self):
        self.bindKey("r", self.rotateLight)
        self.bindKey("R", self.unrotateLight)

        self.bindKey("v", self.addLight)
        
        self.bindKey("x", self.simpleShaderVert)

    def unrotateLight(self):
        self.directionalLights[0]["dir"][1] -= 0.08
        self.rotateLight()
    
    def rotateLight(self):
        a = self.directionalLights[0]["dir"][1]
        a += 0.04
        self.directionalLights[0]["dir"][1] = (a % pi)
        self.directionalLights[1]["dir"][1] = (a % pi) + pi
        ti = abs(self.directionalLights[0]["dir"][1])

        sc = self.shadowCams[0]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -60 * viewVec(*sc["dir"]) + numpy.array([0, 15, 0])
        self.updateShadowCam(0)
        
        sc["bias"] = 0.12 * abs(cos(ti))**2 + 0.03
        self.shadowMap(0, bias=sc["bias"] + 0.05)
        self.simpleShaderVert()
        
        d = self.directionalLights[0]
        self.draw.setPrimaryLight(np.array([d["i"]]), np.array([viewVec(*d["dir"])]))

    def addLight(self):
        p = self.zb[self.H//2, self.W//2] * self.viewVec()
        px = self.zb[self.H//2, self.W//2+2] * self.viewVec()
        px += self.zb[self.H//2, self.W//2+2] * -2/self.scale * self.vVhorz()
        py = self.zb[self.H//2+2, self.W//2] * self.viewVec()
        py += self.zb[self.H//2+2, self.W//2] * 2/self.scale * self.vVvert()

        n = np.cross(px - p, py - p)
        n = n / np.linalg.norm(n)
        n = np.round(n)
        self.spotLights.append({"i":[4.,4,4], "pos":self.pos + p, "vec":n})

    def exportLights(self, f="Lights.txt"):
        a = []
        for b in self.spotLights:
            try: b["pos"] = b["pos"].tolist()
            except: pass
            try: b["vec"] = b["vec"].tolist()
            except: pass
            a.append(b)
        with open(f, "w") as x: x.write(json.dumps(a))

    def importLights(self, f="Lights.txt"):
        a = None
        with open(f) as x: a = json.loads(x.read())
        for b in a:
            b["i"] = [5.,5,5]
        self.spotLights.extend(a)

    def createObjects(self):
        st = time.time()
        print("creating objects")
        
        mpath = "../Minecraft/"

##        self.addVertObject(VertModel, [0,0,0], rot=(0,pi,0),
##                           filename=mpath+"Mansion/White.obj",
##                           texture=mpath+"Mansion/White-RGBA.png",
##                           alpha=mpath+"Mansion/White-RGBA.png",
##                           useShaders="cull",
##                           mc=True, scale=1)

##        self.addVertObject(VertSphere, [0,0,0], n=8, scale=3,
##                           texture="../Assets/Blank.png")
##        
##        self.addVertObject(VertSphere, [6,0,0], n=8, scale=3,
##                           texture="../Assets/Blank.png")
        
##        options = {"texture":"../Assets/Blank.png",
##                   "n":8, "scale":1}
##        for i in range(-20, 20, 3):
##            for j in range(-20, 20, 3):
##                for k in range(0, 12, 3):
##                    c = numpy.array((i, k, j), dtype="float")
##                    self.addVertObject(VertSphere, c, **options)

##        self.addVertObject(VertPlane, [1,0,1],
##                           h2=[1,0,0], h1=[0,0,1], n=1,
##                           texture="../Assets/Blank.png")
        
        self.addVertObject(VertTerrain, [-100, 0, -100],
                       heights="../PyOpenCL26/Terrain.tif",
                       texture="../Assets/Grass.png", scale=1,
                       vertScale=3/6553, vertPow=2, vertMax=50000,
                       uvspread=2)
        self.terrain = self.vertObjects[-1]
##
        treef = "../Models/"
        options = {"filename":treef+"Tree4G/Tree_G.obj", "static":True,
                   "texMode":None, "scale":0.002}
        for i in range(-80, 80, 20):
            for j in range(-80, 80, 20):
                c = numpy.array((i, 0, j), dtype="float")
                c += nr.rand(3) * 8
                c[1] = self.terrain.getHeight(c[0],c[2]) - 0.2
                r = random.random() * 3
                self.addVertObject(VertModel, c, **options, rot=(0,r,0))

        
        self.shadowCams.append({"pos":[0, 10, 0], "dir":[pi/2, 1.1],
                                "size":4096, "scale":90})
        self.shadowCams.append({"pos":[0, 10, 0], "dir":[pi/2, 1.1],
                                "size":512, "scale":15, "gi":1})

        self.pointLights.append({"i":[0.,0,0], "pos":[10,5,10]})
        self.spotLights.append({"i":[0,0,0.], "pos":[20,5,10], "vec":[0,-1,0]})

        self.directionalLights.append({"dir":[pi*1.7, 0.52], "i":[1.6,1.4,1.2]})
        self.directionalLights.append({"dir":[pi*1.7, 0.52+pi], "i":[0.5,0.5,0.5]})
        self.directionalLights.append({"dir":[0, pi/2], "i":[0.1,0.2,0.6]})

        self.makeObjects(0)
        
        #self.skyBox = TexSkyBox(self, 7, "../Skyboxes/Desert_2k.hdr",
        #                        rot=(0,0,0), hdrScale=8)
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Sky_is_on_Fire_4k.hdr",
        #                        rot=(0,pi,0), hdrScale=12)
        #self.skyBox = TexSkyBox(self, 7, "../Skyboxes/Qwantani_4k.hdr",
        #                        rot=(0,pi,0), hdrScale=32)
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Mealie_Road_4k.hdr",
        #                        rot=(0,0,0), hdrScale=16)
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Evening_Road_2k.hdr",
        #                        rot=(0,0,0), hdrScale=8)
        #self.skyBox.created()
        self.skyTex = np.empty((1,6,3), "uint16")

        print("done in", time.time() - st, "s")
        
    def postProcess(self):
        #self.draw.blur()
        self.draw.gamma(1)
        
    def onStart(self):
        self.shadowObjects()
        self.setupShadowCams()
        self.rotateLight()
        self.draw.clearShadowMap(1)

    def frameUpdate(self):
        #self.doQuit = True
        #self.pos[1] = self.terrain.getHeight(self.pos[0], self.pos[2]) + 2
        pass

"""
from Ops import cl, cq
i = np.empty((app.H//16, app.W//16, 128), "int32"); cl.enqueue_copy(cq, i, app.draw.IBUF)
n = np.empty((app.H//16, app.W//16), "int32"); cl.enqueue_copy(cq, n, app.draw.NBUF)
Image.fromarray((n*20).astype("uint8")).save("Test.png")
"""

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
    finally:
        with open("Profile.txt", "w") as f:
            f.write("Backend profile\n\n")
            f.write(app.printProfile())
