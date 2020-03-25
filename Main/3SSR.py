# 3D render
# Multiprocess
# Using PyOpenCL
# Test SSR

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
        super().__init__(width=960, height=640, fovx=90,
                         downSample=1)
        
        self.changeTitle("AXI OpenCL 3D Visualizer")

        # For MWhite
        #self.α = -0.8; self.β = -0.15
        #self.pos = [2.3, 4.5, -5.2]
        #self.α = 5.23; self.β = 0.06
        #self.pos = [12.2, 11.8, -5.9]
        #self.α = 2.773; self.β = 0.1205
        #self.pos = [-4.2971, 3.7685, 8.9507]
        #self.α = 1.54; self.β = -0.14
        #self.pos = [-12.3, 4.8, 0.4]
        self.α = -0.825; self.β = -0.132
        self.pos = [6, 4.61, -7.87]
        #self.α = 5.6; self.β = 0.09
        #self.pos = [0.36, 4.81, -2.76]
        
        self.camSpeed = 0.2
        self.mouseSensitivity = 10

        self.ambLight = 0.001

        self.maxFPS = 24

        self.st = 0
        self.tt = 200
        self.drawAxes = False

        self.camLock = False

    def customizeFrontend(self):
        self.bindKey("r", self.rotateLight)
        self.bindKey("R", self.unrotateLight)

        self.bindKey("g", self.doGI)
        
        self.bindKey("v", self.addLight)
        
        #self.bindKey("c", self.makecubemap)
        self.bindKey("x", self.simpleShaderVert)

    def makecubemap(self):
        S = self.skyTex.shape[1]
        mask = self.reflSphere.texNum
        c = self.reflSphere.coords
        cM = self.draw.drawCubeMap(S, c, self.texUseAlpha, self.texShadow,
                                   self.texMip, self.texRefl,
                                   maskNum=mask)
        cM = CubeMap(cM, 2); a = cM.texture.reshape((-1, 3))
        self.draw.setReflTex("a", a[:,0], a[:,1], a[:,2], cM.m)

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
        
        sc = self.shadowCams[1]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -60 * viewVec(*sc["dir"]) + numpy.array([0, 15, 0])
        self.updateShadowCam(1)
        
        sc["bias"] = 0.12 * abs(cos(ti))**2 + 0.03
        self.shadowMap(0, bias=sc["bias"] + 0.05)
        self.simpleShaderVert()
        
        d = self.directionalLights[0]
        self.draw.setPrimaryLight(np.array([d["i"]]), np.array([viewVec(*d["dir"])]))

    def doGI(self):
        self.draw.clearShadowMap(1)
        self.draw.drawDirectional(1, self.castObjs)

        self.g = self.draw.getGIM(1)
        sc = self.shadowCams[1]
        zb = self.g[1]
        nb = self.g[2]
        vm = viewMat(*sc["dir"])
        #self.spotLights = []
        # Intel can handle 4, 0.02
        # Nvidia needs 10, 0.12
        for i in range(20, sc["size"]-20, 4):
            for j in range(20, sc["size"]-20, 4):
                cz = zb[j, i] - 0.08
                if cz < 6000:
                    p = np.array(sc["pos"], dtype="float") + cz*vm[0]
                    p += (i-sc["size"]/2)/sc["scale"] * vm[1]
                    p += -(j-sc["size"]/2)/sc["scale"] * vm[2]
                    d = nb[j, i][:3]
                    ix = self.g[0][j, i] / 256 * self.directionalLights[0]["i"]
                    self.spotLights.append({"i":0.004 * ix, "pos":p, "vec":-d})

        #Image.fromarray(np.clip(zb*3, 0, 255).astype("uint8"), "L").save("test.png")
        self.draw.clearShadowMap(1)

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

        self.addVertObject(VertModel, [0,0,0], rot=(0,pi,0),
                           filename=mpath+"Mansion/White.obj",
                           texture=mpath+"Mansion/White-RGBA.png",
                           alpha=mpath+"Mansion/White-RGBA.png",
                           #emissive=mpath+"Mansion/White-Emissive_.png",
                           useShaders="cull",
                           mc=True, scale=1, shadow="CR")
        self.shadowCams.append({"pos":[0, 10, 0], "dir":[pi/2, 1.1],
                                "size":6144, "scale":90})
        self.shadowCams.append({"pos":[0, 10, 0], "dir":[pi/2, 1.1],
                                "size":1024, "scale":15, "gi":1})

##        self.addVertObject(VertPlane, [-4, 3.5, -4], n=64,
##                           h2=[12,0,0], h1=[0,0,12],
##                           texture="../Assets/Blank.png")
##        self.addVertObject(VertPlane, [-4, 8.5, -4], n=64,
##                           h1=[12,0,0], h2=[0,0,12],
##                           texture="../Assets/Blank.png")
##        self.addVertObject(VertPlane, [-6, 3.5, 0], n=32,
##                           h2=[4,0,0], h1=[0,4,0],
##                           texture="../Assets/Blank.png")
        self.addVertObject(VertWater, [-10, 3.5, -8], size=180,
                           scale=0.1, pScale=0.04,
                 wDir=[(0.4,0), (0, 0.4)],
                 wLen=[(10, 4, 3), (7, 5, 2)],
                 wAmp=[(0.8, 0.5, 0.3), (0.6, 0.4, 0.3)],
                 wSpd=[(0.6, 0.8, 1.1), (1, 1.1, 1.3)], numW=3,
                           texture="../Assets/Blue.png")
        self.water = self.vertObjects[-1]

        self.pointLights.append({"i":[0.,0,0], "pos":[10,5,10]})
        self.spotLights.append({"i":[0,0,0.], "pos":[20,5,10], "vec":[0,-1,0]})

        self.directionalLights.append({"dir":[pi*1.7, 0.52], "i":[1.6,1.4,1.2]})
        self.directionalLights.append({"dir":[pi*1.7, 0.52+pi], "i":[0.0,0.0,0.0]})
        self.directionalLights.append({"dir":[0, pi/2], "i":[0.1,0.2,0.6]})
        #self.directionalLights.append({"dir":[pi*0.76, 0.8], "i":[0.2,0.1,0.0]})
        
        self.makeObjects(0)
        
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Skybox1a.png",
        #                        rot=(0,-pi/2-0.6,0))
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Desert_2k.hdr",
        #                        rot=(0,pi/3,0), hdrScale=8)
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Sky_is_on_Fire_4k.hdr",
        #                        rot=(0,pi,0), hdrScale=12)
##        self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Qwantani_4k.hdr",
##                                rot=(0,0,0), hdrScale=32)
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Mealie_Road_4k.hdr",
        #                        rot=(0,0,0), hdrScale=16)
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Evening_Road_2k.hdr",
        #                        rot=(0,0,0), hdrScale=8)
##        self.skyBox.created()
        self.skyTex = np.empty((1,6,3), "uint16")

        print("done in", time.time() - st, "s")
        
    def postProcess(self):
        self.draw.blur()
        self.draw.gamma(0.75)
        
    def onMove(self):
        pass
        
    def onStart(self):
##        self.cubeMap = CubeMap(self.skyTex, 2, False)
##        a = self.cubeMap.texture.reshape((-1, 3))
##        self.draw.setReflTex("0", a[:,0], a[:,1], a[:,2], self.cubeMap.m)
##        self.draw.setHostSkyTex(self.cubeMap.rawtexture)
        
        self.shadowObjects()
        self.setupShadowCams()
        self.draw.clearShadowMap(1)
        self.importLights("../PyOpenCL26/LightsAll1.txt")
        self.rotateLight()

    def frameUpdate(self):
        if self.frameNum == 0:
            #self.doGI()
            self.simpleShaderVert()
##        if self.frameNum < 90: pass
##        else: self.doQuit = True
        self.water.update()

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
        print(app.pos, app.facing())
    except:
        raise
    finally:
        with open("Profile.txt", "w") as f:
            f.write("Backend profile\n\n")
            f.write(app.printProfile())
