# 3D render
# Multiprocess
# Using PyOpenCL

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

from VR import VRApp

class ThreeDApp(ThreeDBackend, VRApp):
    def __init__(self):
        super().__init__(width=400, height=400, fovx=120,
                         downSample=1)

        self.setupVR()
        self.VRScale = 1.6
        
        self.changeTitle("AXI OpenCL 3D Visualizer")

##        self.pos = np.array([-1.12, 4.71, 4.25])
##        self.α = 5.5; self.β = 1.3
        self.pos = np.array([3.455, 2.902, 15.339])
        self.lookAt([-4.49, 1.65, 0.95])
##        self.pos = np.array([-2.16, 2.68, 6.01])
##        self.α = 2.51; self.β = 0.3
        
        self.camSpeed = 0.2
        self.mouseSensitivity = 25

        self.ambLight = 0.001

        self.maxFPS = 60

        self.st = 0
        self.tt = 200
        self.drawAxes = False

        self.exptarg = 1

        self.norm = True
        self.dofFoc = 3
        self.doSSAO = True

    def customizeFrontend(self):
        self.bindKey("r", self.rotateLight)
        self.bindKey("R", self.unrotateLight)

        self.bindKey("g", self.doGI)
        self.bindKey("v", self.addLight)

        self.bindKey("f", self.testSpec)
        self.bindKey("F", self.testSpec2)
        
        self.bindKey("x", self.simpleShaderVert)

        self.bindKey("5", self.expDn)
        self.bindKey("6", self.expUp)

        self.bindKey("e", self.tgNorm)
        
        self.bindKey("<Up>", self.z1)
        self.bindKey("<Down>", self.z2)
        
        self.enableDOF(dofR=24, rad=0.03, di=4)

        self.bindKey("f", self.foc1)
        self.bindKey("F", self.foc2)
        
        self.bindKey("c", self.tgAO)
        
    def tgAO(self): self.doSSAO = not self.doSSAO
    def foc1(self): self.dofFoc *= 1.1
    def foc2(self): self.dofFoc /= 1.1

    def testSpec(self):
        self.matShaders[1]["normal"][1] *= 1.1
    def testSpec2(self):
        self.matShaders[1]["normal"][1] /= 1.1
        
    def expDn(self): self.exptarg = max(self.exptarg / 1.1, 0.25)
    def expUp(self): self.exptarg = min(self.exptarg * 1.1, 16)
    def z1(self): self.setFOV(max(30, self.fovX * 0.96))
    def z2(self): self.setFOV(min(120, self.fovX * 1.04166))

    def tgNorm(self):
        self.norm = not self.norm
        if self.norm: self.matShaders[5]["normal"] = ("5", 0.9)
        else: del self.matShaders[5]["normal"]
        
    def doGI(self):
        self.draw.clearShadowMap(1)
        self.draw.drawDirectional(1, self.castObjs)

        self.g = self.draw.getGIM(1)
        sc = self.shadowCams[1]
        zb = self.g[1]
        nb = self.g[2]
        vm = viewMat(*sc["dir"])
        self.spotLights = []
        for i in range(20, sc["size"]-20, 6):
            for j in range(20, sc["size"]-20, 6):
                cz = zb[j, i] - 0.08
                if cz < 6000:
                    p = np.array(sc["pos"], dtype="float") + cz*vm[0]
                    p += (i-sc["size"]/2)/sc["scale"] * vm[1]
                    p += -(j-sc["size"]/2)/sc["scale"] * vm[2]
                    d = nb[j, i][:3]
                    ix = self.g[0][j, i] / 256 * self.directionalLights[0]["i"]
                    self.spotLights.append({"i":0.004 * ix, "pos":p, "vec":-d})

        Image.fromarray(np.clip(zb*3, 0, 255).astype("uint8"), "L").save("Test.png")
        self.draw.clearShadowMap(1)

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
        
        sc["bias"] = 0.12 * abs(cos(ti))**2 + 0.05
        self.shadowMap(0, bias=sc["bias"])
        self.simpleShaderVert()

        self.draw.mipShadowMap(0, r=1)
        self.draw.mipShadowMap(0, r=2)
        
        d = self.directionalLights[0]
        self.draw.setPrimaryLight(np.array([d["i"]]), np.array([viewVec(*d["dir"])]))

        sc = self.shadowCams[2]
        self.updateShadowCam(2)
        self.shadowMap(2, bias=0.15)
        self.draw.mipShadowMap(2, r=1)
        self.draw.mipShadowMap(2, r=2)
        
        sc = self.shadowCams[3]
        self.updateShadowCam(3)
        self.shadowMap(3, bias=0.15)
        self.draw.mipShadowMap(3, r=1)
        self.draw.mipShadowMap(3, r=2)
        
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

        self.addVertObject(VertSphere, [-3,3,3], n=32, scale=1,
                           texture="../Assets/cobblestone/cobblestone_floor_08_diff_2k.png",
                           useShaders={"normal":("1", 0.5)},
                           shadow="C")
        self.addVertObject(VertSphere, [-3,3,5], n=32, scale=1,
                           texture="../Assets/kitchen_wood/kitchen_wood_diff_2k.png",
                           useShaders={"normal":["2", 0.4]},
                           shadow="C")
        self.addVertObject(VertSphere, [-3,3,7], n=32, scale=1,
                           texture="../Assets/ornate-gold/ornate-celtic-gold-albedo.png",
                           useShaders={"normal":("6", 0.4), "metal":1},
                           shadow="C")

        self.addVertObject(VertSphere, [-3,3,9], n=32, scale=1,
                            useShaders={"normal":("test", 0.9)}, shadow="C",
                            texture="../Assets/Sand.png")
        
        self.addVertObject(VertTerrain, [-100, 0, -100], scale=1,
                           heights="../Assets/Terrain.tif",
##                           texture="../Assets/cobblestone/cobblestone_floor_08_diff_1k.png",
##                           texture="../Assets/leafy-grass/leafy-grass-albedo.png",
                           texture="../Assets/mud/red_mud_stones_diff_1k.png",
                           #texAlias="../Assets/Blank.png",
                           vertScale=0.1/6553, vertPow=2, vertMax=50000,
                           uvspread=6, shadow="CR",
                           useShaders={"normal":("3", 0.5),
                                       })#"roughness":"r", "mip":2.})
        
        self.terrain = self.vertObjects[-1]

        treef = "../Models/"
        for i in range(-20, 20, 4):
            for j in range(-20, 20, 4):
                c = np.array((i, 0, j), dtype="float")
                r = nr.rand(3) * np.array([0, pi/2, 0])
                self.addVertObject(VertPlane, c, n=4,
                                   h1=[0,2,0], h2=[2,0,2], rot=r,
                                   useShaders={"cull":1, "normal":("5", 0.8)}, shadow="C",
                                   texture="../Assets/sandstone/sandstone_blocks_05_diff_1k.png",
##                                   texture="../Assets/bricks/large_red_bricks_diff_1k.png",
                                   )
                self.addVertObject(VertPlane, c + np.array([-0.05,0,0.]), n=4,
                                   h2=[0,2,0], h1=[2,0,2], rot=r,
                                   useShaders={"cull":1, "normal":("5", 0.8)}, shadow="C",
                                   texture="../Assets/sandstone/sandstone_blocks_05_diff_1k.png",
##                                   texture="../Assets/bricks/large_red_bricks_diff_1k.png",
                                   )

                self.addVertObject(VertSphere, c + np.array([0,1,0]),
                                   n=16, scale=1,
                                   useShaders={"cull":1, "normal":("4", 0.4)},
                                   shadow="C",
                                   texture="../Assets/rock/rock_05_diff_1k.png",
                                   )
                
##        options = {"filename":treef+"pine/Pine.obj", "static":True,
##                   "texMode":None, "scale":0.2, "shadow":"C"}
##        for i in range(-80, 80, 40):
##            for j in range(-80, 80, 40):
##                c = numpy.array((i, 0, j), dtype="float")
##                c += nr.rand(3) * 8
##                c[1] = self.terrain.getHeight(c[0],c[2]) - 0.4
##                r = random.random() * 3
##                self.addVertObject(VertModel, c, **options, rot=(0,r,0))
        
        self.shadowCams.append({"pos":[0, 10, 0], "dir":[pi/2, 1.1],
                                "size":4096, "scale":64})
        self.shadowCams.append({"pos":[0, 10, 0], "dir":[pi/2, 1.1],
                                "size":512, "scale":8, "gi":1})

        ppos = np.array([6,8,1.])
        qpos = np.array([6,8,3.])
        i = [1.,1,1]#[16.,6,8]
        self.pointLights.append({"i":i, "pos":ppos})
        self.pointLights.append({"i":i, "pos":qpos})
        
        self.shadowCams.append({"pos":ppos, "dir":[-pi/2,0.8],
                                "size":512, "scale":256, "perspective":1})
        self.shadowCams.append({"pos":qpos, "dir":[-pi/2,0.8],
                                "size":512, "scale":256, "perspective":1})


        self.spotLights.append({"i":[0,0,0.], "pos":[20,5,10], "vec":[0,-1,0]})

        #self.directionalLights.append({"dir":[pi*1.7, 0.47], "i":[0.,0,0]})
        self.directionalLights.append({"dir":[pi*1.7, 0.47], "i":[1.2,1.1,1.0]})#[1.2,1.0,0.8]})
        
        self.directionalLights.append({"dir":[pi*1.7, 0.47+pi], "i":[0.,0,0]})#[0.2,0.2,0.2]})
        self.directionalLights.append({"dir":[0, pi/2], "i":[0.01,0.02,0.04]})
        
##        self.directionalLights.append({"dir":[pi*1.7, 0.47], "i":[0.3,0.4,0.6]})
##        self.directionalLights.append({"dir":[pi*1.7, 0.47+pi], "i":[0.1,0.1,0.1]})
##        self.directionalLights.append({"dir":[0, pi/2], "i":[0.1,0.1,0.1]})

        

        #self.skyBox = TexSkyBox(self, 7, "../Skyboxes/Skytest.png",
        #                        rot=(0,0,0))
        #self.skyBox = TexSkyBox(self, 7, "../Skyboxes/Desert_2k.hdr",
        #                        rot=(0,0,0), hdrScale=8)
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Sky_is_on_Fire_4k.hdr",
        #                        rot=(0,pi,0), hdrScale=12)
        self.skyBox = TexSkyBox(self, 9, "../Skyboxes/Qwantani_4k.hdr",
                                rot=(0,pi,0), hdrScale=16)
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Mealie_Road_4k.hdr",
        #                        rot=(0,0,0), hdrScale=16)
        #self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Evening_Road_2k.hdr",
        #                        rot=(0,0,0), hdrScale=8)
        self.skyBox.created()
        #self.skyTex = np.empty((1,6,3), "uint16")

##        self.addVertObject(VertPlane, [-1,-1,0],
##                           h1=[2,0,0], h2=[0,2,0], n=1,
##                           texture="../Assets/Blank1.png",
##                           useShaders={"2d":1, "fog":1})
        
        self.makeObjects(0)

        print("done in", time.time() - st, "s")
        
    def postProcess(self):
##        if self.doSSAO:
##            self.draw.ssao()
##        self.draw.dof(self.dofFoc)
        self.draw.exposure(self.exptarg)
        #self.draw.blur()
        self.draw.tonemap()
        
    def onStart(self):
        self.cubeMap = CubeMap(self.skyTex, 2, False)
        a = self.cubeMap.texture.reshape((-1, 3))
        self.draw.setReflTex("0", a[:,0], a[:,1], a[:,2], self.cubeMap.m)
        
        self.shadowObjects()
        self.setupShadowCams()
        self.rotateLight()
        self.unrotateLight()
        self.draw.clearShadowMap(1)

##        for x in "123456":
##            self.addNrmMap("../Assets/BlankNorm.png", x)
        #s = Image.open("../Assets/cobblestone/cobblestone_floor_08_rough_1k.png")
        #s = Image.open("../Assets/leafy-grass/leafy-grass2-roughness.png")
        s = Image.open("../Assets/mud/red_mud_stones_rough_1k.png")
        self.draw.addRghMap(createMips(
            np.array(s.convert("L").rotate(-90))).astype("uint8"), "r")
        
        self.addNrmMap("../Assets/cobblestone/cobblestone_floor_08_nor_2k.png", "1")
        self.addNrmMap("../Assets/kitchen_wood/kitchen_wood_nor_2k.png", "2")
##        self.addNrmMap("../Assets/leafy-grass/leafy-grass-normal.png", "3", mip=True)
        self.addNrmMap("../Assets/mud/red_mud_stones_nor_1k.png", "3", mip=True)
        self.addNrmMap("../Assets/rock/rock_05_nor_1k.png", "4")
        self.addNrmMap("../Assets/sandstone/sandstone_blocks_05_nor_1k.png", "5")
##        self.addNrmMap("../Assets/bricks/large_red_bricks_nor_1k.png", "5")
        self.addNrmMap("../Assets/ornate-gold/ornate-celtic-gold-normal.png", "6")

        self.addNrmMap("../Assets/WriteN.png", "test")

    def frameUpdate(self):
        if self.VRMode: self.frameUpdateVR()
##        self.pos[0] += 0.01
##        if self.frameNum > 250:
##            self.doQuit = True

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
