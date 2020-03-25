# 3D render
# Multiprocess
# Using PyOpenCL
# Streaming Terrain

from tkinter import *
from math import sin, cos, sqrt, pi
import numpy
import numpy.random as nr
import random

nr.seed(1)
random.seed(1)

import time
from Compute import *
import multiprocessing as mp
import os
import json
from Rig import Rig
import Terrain

def getTerrain(rp, tp):
    doQuit = False
    terr = Terrain.RandTerrain(1)
    while not doQuit:
        try:
            a = rp.get(True, 0.2)
        except: continue
        if a is None:
            doQuit = True
            break
        else:
            ga = terr.getArea(*a[3:])
            tp.put((*a[:3], ga))

class ThreeDApp(ThreeDBackend):
    def __init__(self):
        super().__init__(width=960, height=640, fovx=80,
                         downSample=1)
        
        self.changeTitle("AXI OpenCL 3D Visualizer")

        self.α = 3.1
        self.β = -0.1
        
        self.pos = numpy.array([600.8,  20.4, 600.3])
        self.camSpeed = 0.04

        self.mouseSensitivity = 20

        self.ambLight = 0.08

        self.st = 0
        self.tt = 200
        self.drawAxes = False
        self.b = None

        self.chunksize = 50
        self.chunkscale = 0.8
        self.renderDist = 2
        self.rdextra = 0
        self.numWorkers = 4
        
        S = self.chunksize
        L = self.chunkscale
        
        self.cchunk = [int(self.pos[0]/(S*L) + 0.5),
                       int(self.pos[2]/(S*L) + 0.5)]

        terr = Terrain.RandTerrain(1)

        self.rmax = 50000
        
        self.terr = terr
        
        self.tp = [mp.Queue(16) for i in range(self.numWorkers)]
        self.rp = [mp.Queue(16) for i in range(self.numWorkers)]
        self.workLen = [0 for i in range(self.numWorkers)]

        self.vischunks = [None for i in range((self.renderDist*2)**2)]
        self.reqChunks = []
        self.procChunks = []

        for u in range(self.cchunk[0]-self.renderDist,
                       self.cchunk[0]+self.renderDist):
            for v in range(self.cchunk[1]-self.renderDist,
                           self.cchunk[1]+self.renderDist):
                self.reqChunks.append((u,v))

        for i in range(len(self.reqChunks)):
            a = self.reqChunks[0]
            del self.reqChunks[0]
            self.requestChunk(a, i)

    def requestChunk(self, a, tn):
        S = self.chunksize
        L = self.chunkscale
        
        i = self.workLen.index(min(self.workLen))
        self.rp[i].put((tn, a, [a[0]*S*L, 0, a[1]*S*L],
                        a[0]*S, (a[0]+1)*S+1,
                        a[1]*S, (a[1]+1)*S+1, 1))
        self.workLen[i] += 1

        self.procChunks.append(a)

    def customizeFrontend(self):
        self.bindKey("r", self.rotateLight)
        self.bindKey("t", self.refreshChunks)
        self.bindKey("y", self.printwl)

        self.enableDOF()

    def printwl(self):
        print(self.workLen)
        print(self.b)
        
    def rotateLight(self):
        a = self.directionalLights[0]["dir"][1]
        a += 0.05
        self.directionalLights[0]["dir"][1] = (a % pi)
        self.directionalLights[1]["dir"][1] = (a % pi) + pi
        ti = abs(self.directionalLights[0]["dir"][1])

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
        
        options = {"scale":self.chunkscale, "vertScale":4/6553, "vertPow":2.5,
                   "vertMax":self.rmax, "shadow":"CR", "uvspread":2,
                   "useShaders":"multi"}

        self.terrn = []
        for i in range((self.renderDist*2)**2):
            self.addVertObject(VertTerrain, [0, 0, 0],
                               size=(self.chunksize,self.chunksize),
                               texture="../Assets/Grass.png", texMul=0.9,
                               **options)
            self.terrn.append(self.vertObjects[-1])

        self.addVertObject(VertPlane, [0,0,0], n=1, h1=[1,0,0], h2=[0,0,1],
                           texture="../Assets/Rock.png")
        self.addVertObject(VertPlane, [0,0,0], n=1, h1=[1,0,0], h2=[0,0,1],
                           texture="../Assets/Snow.png")
        self.addVertObject(VertPlane, [0,0,0], n=1, h1=[1,0,0], h2=[0,0,1],
                           texture="../Assets/Sand1.png")
        
        self.directionalLights.append({"dir":[pi*2/3, 2.1], "i":[1.4,1.2,0.8]})
        self.directionalLights.append({"dir":[pi*2/3, 2.1+pi], "i":[0.1,0.2,0.1]})
        self.directionalLights.append({"dir":[0, pi/2], "i":[0.1,0.2,0.5]})
        
        self.shadowCams.append({"pos":[40, 5, 40], "dir":[pi/2, 1.1],
                                "size":4096, "scale":48})
        self.shadowCams.append({"pos":[40, 5, 40], "dir":[pi/2, 1.1],
                                "size":1024, "scale":128})
        self.makeObjects(0)
        
##        self.skyBox = TexSkyBox(self, 12, "../Skyboxes/Autumn_Park_4k.hdr",
##                                rot=(0,0,0), hdrScale=16)
##        self.skyBox.created()
        self.skyTex = np.empty((1,6,3), "uint16")

        print("done in", time.time() - st, "s")
        
    def postProcess(self):
        #self.draw.exposure(1)
        #self.draw.fog()
        self.draw.gamma(1)
        
    def onMove(self): pass
    def onStart(self):
        self.tget = []
        for i in range(self.numWorkers):
            bargs = (self.rp[i], self.tp[i])
            self.tget.append(mp.Process(target=getTerrain, args=bargs))
            self.tget[i].start()
        
        self.shadowObjects()
        self.setupShadowCams()
        self.rotateLight()
        #t = createMips(getTexture1("../Assets/Sandstone.png")*0.9)
        #a = self.draw.addTexture(t[:,0], t[:,1], t[:,2])
        #self.texMip[0] = a
        
    def frameUpdate(self):
        if all([i == 0 for i in self.workLen]):
            self.refreshChunks()
        
        for i in range(len(self.tp)):
            if not self.tp[i].empty():
                a = self.tp[i].get(True, 0.2)
                self.workLen[i] -= 1
                #print("get", a[:3])
                self.vischunks[a[0]] = a[1]
                t = VertTerrain(self, a[2],
                                heights=a[3].T, texture="../Assets/Grass.png",
                                scale=self.chunkscale, vertScale=4/6553,
                                vertPow=2, vertMax=self.rmax,
                                shadow="CR", uvspread=4)
                t.create()
                testCr(t)
                self.draw.copyTo(self.terrn[0].texNum, t.vertPoints, t.vertNorms,
                                 cstart=self.terrn[a[0]].cStart)

                #self.shadowMap(0, bias=0.05)
                self.simpleShaderVert()

                if a[1] in self.procChunks:
                    del self.procChunks[self.procChunks.index(a[1])]

    def refreshChunks(self):
        S = self.chunksize
        L = self.chunkscale
        ppos = self.pos # + 8 * self.speed

        self.cchunk = [int(ppos[0]/(S*L) + 0.5),
                       int(ppos[2]/(S*L) + 0.5)]

        for u in range(self.cchunk[0]-self.renderDist+self.rdextra,
                       self.cchunk[0]+self.renderDist-self.rdextra):
            for v in range(self.cchunk[1]-self.renderDist+self.rdextra,
                           self.cchunk[1]+self.renderDist-self.rdextra):
                if ((u, v) not in self.vischunks) and \
                   ((u, v) not in self.reqChunks) and \
                   ((u, v) not in self.procChunks):
                    self.reqChunks.append((u,v))

        #print("req:", self.reqChunks)
        
        if len(self.reqChunks) > 0:
            a = list(range(len(self.vischunks)))
            kf = lambda i: (self.vischunks[i][0] - self.cchunk[0])**2 + \
                           (self.vischunks[i][1] - self.cchunk[1])**2
            df = lambda i: (i[0] - self.cchunk[0])**2 + \
                           (i[1] - self.cchunk[1])**2 
            try:
                a = sorted(a, key=kf, reverse=True)
                self.b = [kf(x) for x in a]
            except:
                random.shuffle(a)
            
            for i in a:
                v = self.vischunks[i]
                if len(self.reqChunks) > 0:
                    a = self.reqChunks[0]
                    del self.reqChunks[0]
                    #print("unload", v, "dist", df(v), "load", a)
                    self.requestChunk(a, i)

def testCr(self):
    wp = np.array(self.wedgePoints)
    s, c = self.scale, self.coords
    self.vertPoints = ne.evaluate("wp * s + c").reshape((-1, 3))
    self.vertNorms = np.array(self.vertNorms).reshape((-1, 3))

if __name__ == "__main__":
    app = ThreeDApp()
    app.start()
    app.runBackend()
    for i in range(4):
        app.rp[i].put(None)
    for i in range(4):
        app.tget[i].join()
    app.finish()
    print("avg spf:", app.totTime / app.frameNum)
    with open("Profile.txt", "w") as f:
        f.write("Backend profile\n\n")
        f.write(app.printProfile())
