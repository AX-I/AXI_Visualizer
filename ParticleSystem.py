# Particle System for 3D visualizer

import numpy as np
import numpy.random as nr
from Utils import anglesToCoords, viewVec
from math import sin, cos, pi
import numexpr as ne

def vVvert(a, b):
    b2 = b - pi/2
    v = np.array([sin(a) * cos(b2),
                  -sin(b2),
                  cos(a) * cos(b2)])
    return -v
def vVhorz(a, b):
    a2 = a + pi/2
    v = np.array([sin(a2), 0, cos(a2)])
    return -v

def rotMat2(a, b):
    rotX = np.array([[1, 0, 0],
                     [0, cos(b), -sin(b)],
                     [0, sin(b), cos(b)]])
    rotY = np.array([[cos(a), 0, sin(a)],
                     [0, 1, 0],
                     [-sin(a), 0, cos(a)]])
    return rotX @ rotY

class ParticleSystem:
    """Single emission"""
    def __init__(self, emitPos, emitDir, size=2,
                 opacity=0.6, color=(255, 255, 255),
                 vel=0.5, nParticles=1000,
                 randPos=0.5, randVel=0.2, randDir=0.1,
                 force=(0, 0, 0), drag=0.001,
                 lifespan=10, colorOverLife=None,
                 randColor=15, tex=None, shSize=2,
                 **ex):
        
        self.N = nParticles
        self.tex = tex
        self.shSize = shSize
        
        self.pc = []
        self.pv = []
        self.opacity = opacity
        self.c1 = np.array(color)
        self.c2 = np.array(color)
        if not (colorOverLife is None):
            self.c2 = np.array(colorOverLife)
        self.color = np.repeat(np.expand_dims(self.c1, 0), self.N, axis=0)
        self.randColor = randColor
        
        self.size = size
        
        self.pos = np.array(emitPos)
        self.emitDir = np.array(emitDir) + (0.5-nr.rand(2))*randDir
        self.vel = vel
        self.dv = anglesToCoords(emitDir) * vel
        self.force = np.array(force)
        self.drag = drag

        self.randPos = randPos
        self.randVel = randVel
        
        self.L = lifespan
        
    def setup(self):
        self.ll = 0
        self.color = np.array(np.repeat(np.expand_dims(self.c1, 0), self.N, axis=0))
        self.color += (nr.randn(self.N, 3) * self.randColor).astype("int")
        self.color = np.clip(self.color, 0, 255)
        
        self.pc = (nr.randn(self.N, 3)) * self.randPos + self.pos
        self.pv = (nr.randn(self.N, 3)) * self.randVel + self.dv
    
    def step(self):
        if (self.dv > 0).any() or self.randVel > 0:
            self.pc += self.pv
            self.pv *= (1-self.drag)
        self.pv += self.force
        test = np.expand_dims(self.c1 * np.clip((self.L-self.ll)/self.L, 0, 1), 0)

        self.color = np.repeat(test, self.N, axis=0)
        self.color += np.repeat(np.expand_dims(self.c2 * np.clip(self.ll / self.L, 0, 1), 0), self.N, axis=0)
        
        self.ll += 1

        self.pos += self.dv

    def reset(self):
        del self.pc, self.pv, self.color
        self.setup()
        
class ContinuousParticleSystem(ParticleSystem):
    """Continous emission"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if (self.N % self.L) > 0:
            raise ValueError("# must be divisible by lifespan!")

    def setup(self):
        super().setup()
        self.pl = np.arange(-self.L, 0, self.L/self.N)
        self.pe = [True] * (self.N // self.L)
        self.pe.extend([False] * (self.N - self.N // self.L))
        self.pe = np.array(self.pe)
        self.Rpc = np.array(self.pc)
        self.Rpv = np.array(self.pv)
        self.pc = self.Rpc[self.pe]
        self.pv = self.Rpv[self.pe]

        self.Rpcolor = np.full((self.N, 3), self.c1)

    def step(self):
        self.pl += 1
        self.Rpc[self.pe] += self.Rpv[self.pe]
        self.Rpv[self.pe] *= (1-self.drag)
        self.Rpv[self.pe] += self.force
        
        self.Rpcolor = self.c1 * np.expand_dims(
            np.clip((self.L-self.pl)/self.L, 0, 1), 1)
        self.Rpcolor += self.c2 * np.expand_dims(
            np.clip(self.pl / self.L, 0, 1), 1)

        self.ll += 1

        self.pc = self.Rpc[self.pe]
        self.pv = self.Rpv[self.pe]
        self.color = self.Rpcolor[self.pe]

        self.pe = (self.pl > 0) & (self.pl < self.L)

    def changeDir(self, newDir):
        self.dv = anglesToCoords(newDir) * self.vel
        self.Rpv[self.pl < 0] = (0.5 - nr.randn(np.sum(self.pl < 0), 3)) * self.randVel + self.dv
    def changePos(self, newpos):
        self.Rpc[self.pl < 0] += newpos - self.pos
        self.pos[:] = newpos
    def reset(self):
        del self.pc, self.pv, self.color, self.pe, self.pl, self.Rpcolor
        self.setup()
        self.step()

class CentripetalParticleSystem(ParticleSystem):
    """Centripetal system, single emission
f => centripetal force, r => distance from center, cc => 1 or -1"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = kwargs["f"]
        self.r = kwargs["r"]
        self.cc = kwargs["cc"]

    def setup(self):
        super().setup()
        self.ll = 0
        
        self.pr = (0.5 - nr.randn(self.N)) * self.randPos + self.r
        self.pa = nr.rand(self.N) * 2*pi
        self.Rpc = np.stack((np.sin(self.pa), np.zeros((self.N,)), np.cos(self.pa))).T
        self.Rpc = self.Rpc * np.expand_dims(self.pr, 0).T

        #self.color = np.zeros((self.N, 3))
        #self.color = self.c1 * np.expand_dims(np.clip(-np.abs(self.r - self.pr)/2+1, 0, 1), 1)
        #self.color += self.c2 * np.expand_dims(np.clip(self.r - self.pr, 0, None), 1)
        
        
        self.pc = self.Rpc @ rotMat2(self.emitDir[0], self.emitDir[1]) + self.pos

        ta = self.pa + self.cc * pi/2
        self.pv = np.stack((np.sin(ta), np.zeros((self.N,)), np.cos(ta))).T
        self.pv = self.pv * np.sqrt(np.abs(self.f / np.expand_dims(self.pr, 0).T))
        self.pv[:,1] += self.vel
        
    def step(self):
        self.Rpc += self.pv

        vn = self.Rpc
        xz = np.stack((vn[:,0], vn[:,2]), axis=1)
        di = np.linalg.norm(xz, axis=1)
        vn[:,0] = xz[:,0] / di * self.pr
        vn[:,2] = xz[:,1] / di * self.pr
        
        rm = np.array([[0, -1], [1, 0]]) * self.cc
        ta = np.stack((vn[:,0], vn[:,2]), axis=1) @ rm

        self.pv = ta * np.sqrt(np.abs(self.f / np.expand_dims(self.pr, 0).T))
        self.pv = np.stack((self.pv[:,0], np.zeros((self.N,)).T, self.pv[:,1]), 1)
        self.pv[:,1] = self.vel

        self.Rpc = vn

        self.pc = self.Rpc @ rotMat2(self.emitDir[0], self.emitDir[1]) + self.pos

    def reset(self):
        self.setup()


class SpiralParticleSystem(ParticleSystem):
    """Centripetal system, continuous emission
f => centripetal force, r => distance from center, cc => 1 or -1"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = kwargs["f"]
        self.r = kwargs["r"]
        self.cc = kwargs["cc"]
        self.turns = kwargs["turns"]
        self.offset = kwargs["offset"]

        if (self.N % self.L) > 0:
            raise ValueError("# must be divisible by lifespan!")

        self.emitDir[0] += pi/2
        self.emitDir[1] = pi/2-self.emitDir[1]

    def setup(self):
        super().setup()

        self.pl = np.arange(0, -self.L, -self.L/self.N, dtype="float")
        self.pe = np.full((self.N,), False)
        self.pe = np.array(self.pe)

        self.Rpcolor = np.full((self.N, 3), self.c1)
        
        self.ll = 0
        
        self.pr = np.full((self.N,), self.r)
        self.pa = np.arange(self.turns*2*pi, step=self.turns*2*pi/self.N)
        self.pa = self.pa[:self.N]
        self.pa += self.offset * 2*pi
        self.pa += (0.5 - nr.randn(self.N)) * self.randPos
        
        y = np.arange(0, -self.vel*self.L, -self.vel*self.L/self.N)
        self.Rpc = np.stack((np.sin(self.pa), y, np.cos(self.pa))).T
        self.Rpc = self.Rpc * np.expand_dims(self.pr, 0).T
        
        self.Spc = self.Rpc @ rotMat2(self.emitDir[0], self.emitDir[1]) + self.pos

        ta = self.pa + self.cc * pi/2
        self.pv = np.stack((np.sin(ta), np.zeros((self.N,)), np.cos(ta))).T
        self.pv = self.pv * np.sqrt(np.abs(self.f / np.expand_dims(self.pr, 0).T))
        self.pv[:,1] = self.vel

        self.pc = self.Spc[self.pe]
        
    def step(self):
        self.Rpc += self.pv

        vn = np.array(self.Rpc)
        xz = np.stack((vn[:,0], vn[:,2]), axis=1)
        di = np.linalg.norm(xz, axis=1)
        vn[:,0] = xz[:,0] / di * self.pr
        vn[:,2] = xz[:,1] / di * self.pr
        
        rm = np.array([[0, -1], [1, 0]]) * self.cc
        ta = np.stack((vn[:,0], vn[:,2]), axis=1) @ rm
        f = self.f
        pr = np.expand_dims(self.pr, 1)
        self.pv = ne.evaluate("ta * sqrt(abs(f / pr))")
        self.pv = np.stack((self.pv[:,0],
                            np.full_like(self.pv[:,0], self.vel),
                            self.pv[:,1]), 1)

        self.Rpc = np.array(vn)
        self.Spc = self.Rpc @ rotMat2(self.emitDir[0], self.emitDir[1]) + self.pos
        self.pc = self.Spc[self.pe]

        self.pl += 1/self.r
        self.ll += 1

        self.Rpcolor = self.c1 * np.expand_dims(
            np.clip((self.L-self.pl)/self.L, 0, 1), 0).T
        self.Rpcolor += self.c2 * np.expand_dims(
            np.clip(self.pl / self.L, 0, 1), 0).T
        self.color = self.Rpcolor[self.pe]

        self.pe = (self.pl > 0) & (self.pl < self.L)

    def reset(self):
        self.setup()

if __name__ == "__main__":
    a = SpiralParticleSystem([0, 0, 0], [0, 0.5], nParticles=4,
                             randPos=0, randVel=0,
                             f=1, r=1, cc=1)
    a.setup()
    a.step()
