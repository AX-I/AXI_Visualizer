# 3D render
# Multiprocess
# Using PyOpenCL
# Global Illumination
# Minecraft models

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
        super().__init__(width=1280, height=800, fovx=75,
                         downSample=1)
        # Resolutions
        # 3840 x 2160  (16:9)
        # 2880 x 2160  (4:3)
        # 2400 x 1800  (4:3)
        # 1920 x 1080  (16:9) (fs)
        # 1280 x 1024  (5:4)  (for Freefall)
        # 1280 x 800   (8:5)  (for C-build)
        # 1280 x 720   (16:9) (fs)
        # 1200 x 900   (4:3)
        
        self.changeTitle("AXI OpenCL 3D Visualizer")

        #self.α = 8.95
        #self.β = -0.322
        #self.pos = numpy.array([-25.25,  10.53, 52.09])
        self.α = 1.11
        self.β = 0.05
        self.pos = numpy.array([-59.35,  25.83, -42.29])
        
        self.camSpeed = 0.2
        self.mouseSensitivity = 10

        self.ambLight = 0.001

        self.maxFPS = 24

        self.st = 0
        self.tt = 200
        self.drawAxes = False

        self.camLock = False

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
        self.test = cM
        cM = CubeMap(cM, 2)
        a = cM.texture.reshape((-1, 3))
        self.draw.setReflTex("Earth", a[:,0], a[:,1], a[:,2], cM.m)
        
    def rotateLight(self):
        a = self.directionalLights[0]["dir"][1]
        a += 0.04
        self.directionalLights[0]["dir"][1] = (a % pi)
        self.directionalLights[1]["dir"][1] = (a % pi) + pi
        ti = abs(self.directionalLights[0]["dir"][1])

        #self.ambLight = 0.1 * sin(ti) + 0.05
        sc = self.shadowCams[0]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -60 * viewVec(*sc["dir"]) + numpy.array([0, 15, 0])
        self.updateShadowCam(0)
        sc["bias"] = 0.12 * abs(cos(ti))**2 + 0.03
        self.shadowMap(0, bias=sc["bias"])
        self.simpleShaderVert()
        
        d = self.directionalLights[0]
        self.draw.setPrimaryLight(np.array([d["i"]]), np.array([viewVec(*d["dir"])]))

    def doGI(self):
        self.draw.clearShadowMap(0)
        self.draw.drawDirectional(0, self.castObjs)

        #d = np.array(self.directionalLights[0]["i"])

        self.g = self.draw.getGIM(0)
        sc = self.shadowCams[0]
        zb = self.g[1]
        nb = self.g[2]
        vm = viewMat(*sc["dir"])
        self.spotLights = []
        for i in range(40, sc["size"]-40, 8):
            for j in range(40, sc["size"]-40, 8):
                cz = zb[j, i] - 0.08
                if cz < 6000:
                    p = np.array(sc["pos"], dtype="float") + cz*vm[0]
                    p += (i-sc["size"]/2)/sc["scale"] * vm[1]
                    p += -(j-sc["size"]/2)/sc["scale"] * vm[2]
                    d = nb[j, i][:3]
                    ix = self.g[0][j, i] / (2**8) / 255
                    self.spotLights.append({"i":0.34 * ix, "pos":p, "vec":-d})
        #print(len(self.spotLights))
        #self.simpleShaderVert()
        #Image.fromarray((self.g[0]>>8).astype("uint8")).save("test.png")

    def createObjects(self):
        st = time.time()
        print("creating objects")
        
        mpath = "../Models/"

        self.addVertObject(VertSphere, [33, 6, 26], n=48, scale=2,
                           texture="../Assets/Earth2a.png",
                           reflect="Earth",
                           rot=(0, 2*pi/3, 0), shadow="C")
        self.reflSphere = self.vertObjects[-1]

        self.addVertObject(VertModel, [0,0,0],
                           filename="../Minecraft/a.obj",
                           texture="../Minecraft/a-RGBA.png",
                           alpha="../Minecraft/a-RGBA.png",
                           mc=True, scale=1, shadow="CR")
        self.shadowCams.append({"pos":[0, 15, 0], "dir":[pi/2, 1.1],
                                "size":8192, "scale":62, "gi":1})

##        self.addVertObject(VertModel, [0,-50,0],
##                           filename="../Minecraft/F.obj",
##                           texture="../Minecraft/a-RGBA.png",
##                           alpha="../Minecraft/a-RGBA.png",
##                           mc=True, scale=1, shadow="CR")
##        self.shadowCams.append({"pos":[0, 15, 0], "dir":[pi/2, 1.1],
##                                "size":6144, "scale":72, "gi":1})

        self.pointLights.append({"i":[0.2,0,0], "pos":[10,5,10]})
        self.spotLights.append({"i":[0,0,0.5], "pos":[20,5,10], "vec":[0,-1,0]})

        self.directionalLights.append({"dir":[pi*2/3, 0.4], "i":[1.8,1.5,0.7]})
        self.directionalLights.append({"dir":[pi*2/3, 0.4+pi], "i":[0.0,0.0,0.0]})
        self.directionalLights.append({"dir":[0, pi/2], "i":[0.1,0.2,0.4]})

        self.shadowCams.append({"pos":[40, 5, 40], "dir":[pi/2, 1.1],
                                "size":1024, "scale":128})
        
        self.makeObjects(0)
        
        self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Skybox1.png",
                                rot=(0,-pi/2-0.6,0))
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
        pass

if __name__ == "__main__":
    mp.set_start_method("forkserver")
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
