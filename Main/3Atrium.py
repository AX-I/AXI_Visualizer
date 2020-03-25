# 3D render
# Multiprocess
# Using PyOpenCL

# Atrium

from tkinter import *
from math import sin, cos, sqrt, pi, atan2
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

        #self.recVideo = True

        # For Atrium
        self.α = 5.576; self.β = -0.581
        self.pos = np.array([19.0341,  0.9633, -7.6588])
        #self.α = 0.606; self.β = 0.015
        #self.pos = np.array([11.566,  2.365,  8.682])
        
        self.camSpeed = 0.2
        self.mouseSensitivity = 10

        self.ambLight = 0.001

        self.maxFPS = 24

        self.st = 0
        self.tt = 200
        self.drawAxes = False

        self.text1 = None
        self.exptarg = 1

    def customizeFrontend(self):
        self.bindKey("r", self.rotateLight)
        self.bindKey("R", self.unrotateLight)

        self.bindKey("g", self.doGI)

        self.bindKey("v", self.addLight)
        
        #self.bindKey("x", self.simpleShaderVert)

        self.bindKey("5", self.expDn)
        self.bindKey("6", self.expUp)

        self.enableDOF(dofR=15, rad=0.1, di=3)

    def expDn(self): self.exptarg = max(self.exptarg / 1.1, 0.25)
    def expUp(self): self.exptarg = min(self.exptarg * 1.1, 16)

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
        sc["pos"] = -80 * viewVec(*sc["dir"]) + numpy.array([0, 15, 0])
        self.updateShadowCam(0)
        
        sc = self.shadowCams[1]
        sc["dir"] = self.directionalLights[0]["dir"]
        sc["pos"] = -80 * viewVec(*sc["dir"]) + numpy.array([0, 15, 0])
        self.updateShadowCam(1)
        
        sc["bias"] = 0.12 * abs(cos(ti))**2 + 0.03
        self.shadowMap(0, bias=sc["bias"] + 0.04)
        self.simpleShaderVert()
        
        d = self.directionalLights[0]
        self.draw.setPrimaryLight(np.array([d["i"]]), np.array([viewVec(*d["dir"])]))

    def doGI(self):
        self.draw.clearShadowMap(1)
        self.draw.drawDirectional(1, self.castObjs, self.texUseAlpha)

        self.spotLights = []

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
                    # 0.004 for MW, 0.02 for C, 0.009 for R
                    self.spotLights.append({"i":0.009 * ix, "pos":p, "vec":-d})
                    #self.spotLights.append({"i":0.02 * ix, "pos":p, "vec":-d})

        Image.fromarray(np.clip(zb*3, 0, 255).astype("uint8"), "L").save("test.png")
        self.draw.clearShadowMap(1)

    def addLight(self):
        p = self.zb[self.H//2, self.W//2] * self.viewVec()
##        self.pointLights.append({"i":[4.,4,4], "pos":self.pos + p})
        
        px = self.zb[self.H//2, self.W//2+2] * self.viewVec()
        px += self.zb[self.H//2, self.W//2+2] * -2/self.scale * self.vVhorz()
        py = self.zb[self.H//2+2, self.W//2] * self.viewVec()
        py += self.zb[self.H//2+2, self.W//2] * 2/self.scale * self.vVvert()

        n = np.cross(px - p, py - p)
        n = n / np.linalg.norm(n)
        n = np.round(n)
        self.spotLights.append({"i":[4.,4,4], "pos":self.pos + p, "vec":n})

    def importLights(self, f):
        a = None
        with open(f) as x: a = json.loads(x.read())
        for b in a:
            b["i"] = [0.1,0.1,0.1]
            
            b["vec"] = np.array(b["vec"]).tolist()
            #b["pos"][0] *= -1; b["pos"][2] *= -1
            
        self.spotLights.extend(a)
        #self.pointLights.extend(a)

    def createObjects(self):
        st = time.time()
        print("creating objects")
        
        self.importLights("../PyOpenCL26/LightsCB.txt")

        self.addVertObject(VertModel, [0,0,0], rot=(0,0,0),
                           filename="../Atrium/Atrium4.obj",
                           #filename="Test4.obj",
                           scale=1,
                           useShaders="cull",
                           subDiv=3,
                           blender=True)
        
        self.test = self.vertObjects[-1]
        
        self.shadowCams.append({"pos":[0, 10, 0], "dir":[pi/2, 1.1],
                                "size":2048, "scale":90})
        self.shadowCams.append({"pos":[0, 10, 0], "dir":[pi/2, 1.1],
                                "size":1024, "scale":15, "gi":1})

##        self.lSpheres = []
##        self.lTime = []
##        for i in range(0, 21, 7):
##            for j in range(-10, 10, 7):
##                self.lTime.append(random.random() * 5)
##                self.addVertObject(VertSphere, [i, 2 + sin(self.lTime[-1]), j], scale=0.1, n=8,
##                                   texture="../Assets/Blank.png",
##                                   alpha="../Assets/Blank.png")
##                self.lSpheres.append(self.vertObjects[-1])
##                self.pointLights.append({"i":[2.,2,2], "pos":[i,2 + sin(self.lTime[-1]),j]})
        self.spotLights.append({"i":[0,0,0.], "pos":[20,5,10], "vec":[0,-1,0]})

        self.directionalLights.append({"dir":[0, 0], "i":[0,0,0.]})
        self.directionalLights.append({"dir":[0, 0], "i":[0,0,0.]})
        self.directionalLights.append({"dir":[0, 0], "i":[0,0,0.]})
        self.directionalLights.append({"dir":[0, -pi/2], "i":[0.2,0.2,0.2]})
        #self.directionalLights.append({"dir":[pi*0.7, 0.52], "i":[1.6,1.4,1.2]})
        #self.directionalLights.append({"dir":[pi*1.7, 0.52], "i":[0.8,0.7,0.6]})
        
        self.directionalLights.append({"dir":[pi*1.7, 0.52], "i":[1.2,0.9,0.7]})
        self.directionalLights.append({"dir":[pi*1.7, 0.52+pi], "i":[0.4,0.4,0.4]})
        self.directionalLights.append({"dir":[0, pi/2], "i":[0.1,0.2,0.4]})
##        self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Qwantani_4k.hdr",
##                                rot=(0,0,0), hdrScale=20)
        
##        self.directionalLights.append({"dir":[pi*1.7, 0.2], "i":[1.3,0.9,0.7]})
##        self.directionalLights.append({"dir":[pi*1.7, 0.2+pi], "i":[0.3,0.2,0.1]})
##        self.directionalLights.append({"dir":[0, pi/2], "i":[0.25,0.15,0.35]})
##        self.directionalLights.append({"dir":[pi*0.76, 0.6], "i":[0.1,0.12,0.15]})
##        self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Sky_is_on_Fire_4k.hdr",
##                                rot=(0,0,0), hdrScale=10)
        
        self.makeObjects(0)
        
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Desert_2k.hdr",
        #                        rot=(0,pi/3,0), hdrScale=8)
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Mealie_Road_4k.hdr",
        #                        rot=(0,0,0), hdrScale=16)
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Evening_Road_2k.hdr",
        #                        rot=(0,0,0), hdrScale=8)
##        self.skyBox.created()
        self.skyTex = np.empty((1,6,3), "uint16")

        print("done in", time.time() - st, "s")
        
    def postProcess(self):
        #self.draw.exposure(self.exptarg)
        #self.draw.blur()
        self.draw.gamma(1)
        
    def onStart(self):
        #self.renderMask = [x not in [22] for x in range(len(self.vtNames))]
##        self.cubeMap = CubeMap(self.skyTex, 2, False)
##        a = self.cubeMap.texture.reshape((-1, 3))
##        self.draw.setReflTex("a", a[:,0], a[:,1], a[:,2], self.cubeMap.m)
##        self.draw.setReflTex("0", a[:,0], a[:,1], a[:,2], self.cubeMap.m)
##        self.draw.setHostSkyTex(self.cubeMap.rawtexture)
        
        self.shadowObjects()
        self.setupShadowCams()
        #self.rotateLight()
        self.draw.clearShadowMap(1)

##        self.exptarg = 1.2

        self.simpleShaderVert()
        
##        self.pos = np.array([-43.71, 8.914, 21.253])
##        self.lookAt(np.array([37.017, 12.91, -17.962]))
##        self.exptarg = 0.3957

    def frameUpdate(self):
        pass
##        for i in range(len(self.lSpheres)):
##            p = self.lSpheres[i]
##            s = 0.1 * cos(self.frameNum * 0.1 + self.lTime[i])
##            self.draw.translate(np.array((0,s,0)), p.cStart*3, p.cEnd*3, p.texNum)
##            self.pointLights[i]["pos"][1] = 2 + sin(self.frameNum * 0.1 + self.lTime[i])
##        self.simpleShaderVert()

    def exportObj(self, fn):
        mtln = ".".join(fn.split(".")[:-1]) + ".mtl"

        from ObjExport import exportObjMultiTexNorm
        c = self.test
        k = list(self.vtNames.keys())
        self.ctexn = range(len(self.vertPoints))

        exportObjMultiTexNorm(fn, mtln, [k[i] for i in self.ctexn],
                               [self.vertPoints[i] for i in self.ctexn],
                               [np.stack((self.vertU[i],
                                          self.vertV[i]), axis=2)
                                for i in self.ctexn],
                               [self.vertNorms[i] for i in self.ctexn],
                               mtlAlias=mtln.split("/")[-1])


if __name__ == "__main__":
    app = ThreeDApp()
    self = app
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
