# OpenCL rasterization
# Multiple rigs

import numpy as np
import time
from Utils import viewMat

import pyopencl as cl

#ctx = cl.Context(cl.get_platforms()[0].get_devices())

import os
os.environ["PYOPENCL_CTX"] = "0"

ctx = cl.create_some_context()
print("Using", ctx.devices[0].name)
cq = cl.CommandQueue(ctx)
mf = cl.mem_flags

BLOCK_SIZE = 256

def makeProgram(f, path="Shaders/"):
    global ctx
    t = open(path + f).read().replace("BLOCK_SIZE", str(BLOCK_SIZE))
    return cl.Program(ctx, t).build()

d = ctx.devices[0]
PLATFORM = None
if ("Intel" in d.name) or ("Intel" in d.vendor):
    PLATFORM = "intel"
elif ("GeForce" in d.name) or ("NVIDIA" in d.vendor):
    PLATFORM = "nvidia"
elif ("AMD" in d.name) or ("AMD" in d.vendor):
    PLATFORM = "amd"

VECTORIZED = d.native_vector_width_float > 1

vert = makeProgram("vert.c", "Pipe/")
trisetup = makeProgram("trisetup_original.c", "Pipe/")
gather = makeProgram("gather.c", "Pipe/")

draw = makeProgram("drawtexcolsmshplerp.c")
drawA = makeProgram("drawtexcolsmshalphaplerp_max.c")
#drawTri = makeProgram("drawtexcolsmshplerpmiptri.c")
#drawTriS = makeProgram("drawtexcolsmsh2plerpmiptri_xor.c")
drawTriS = makeProgram("drawtexcolsmsh2plerpmiptri_max.c")
drawTriSOp = makeProgram("drawtexcolsmsh2plerpmiptri_maxop.c")
#drawCM = makeProgram("drawreflcubemap.c")
#drawCMF = makeProgram("drawreflcubemapfresnel.c")
drawCMT = makeProgram("drawreflcubemapfresneltexsh_cc.c")

drawNorm = makeProgram("drawtexnorm.c")
drawAdd = makeProgram("drawadd.c")

blur1 = makeProgram("Post/blur.c")
blur2 = makeProgram("Post/blur2.c")

wave = makeProgram("wave2.c", "VertShaders/")
skel = makeProgram("bone1.c", "VertShaders/")

sct = makeProgram("select.c", "VertShaders/")

prg2 = makeProgram("drawskylerp.c")

sun = makeProgram("drawsun.c")

trisetupOrtho = makeProgram("trisetupOrtho.c", "Pipe/")
sh = makeProgram("drawmin.c")
shA = makeProgram("drawminalpha.c")
clearzb = makeProgram("clearzb.c")

gamma = makeProgram("Post/gamma.c")

particles = makeProgram("ps.c")
particlestex = makeProgram("pstranstex.c")
cloud = makeProgram("cloud.c")
cloudsh = makeProgram("cloudshadow.c")
cloudshop = makeProgram("cloudshadowopacity.c")

def makeRBuf(nbytes):
    return cl.Buffer(ctx, mf.READ_ONLY, size=nbytes)

def align34(a):
    return np.stack((a[:,0], a[:,1], a[:,2], np.zeros_like(a[:,0])), axis=1)

class CLDraw:
    def __init__(self, max_s, size_sky, max_uv, w, h, max_particles):
        self.W = np.int32(w)
        self.H = np.int32(h)
        self.A = w*h

        sps = np.ones((max_s*3, 2), dtype="int32")
        us = np.zeros((max_s*3,), dtype="float32")
        rsi = np.ones((size_sky*size_sky,), dtype="uint16")
        
        ro = np.ones((h, w), dtype="uint16")
        db = np.full((h, w), 255, dtype="float32")
        
        self.SPS = cl.Buffer(ctx, mf.READ_ONLY, size=sps.nbytes)
        self.DB = cl.Buffer(ctx, mf.READ_WRITE, size=db.nbytes)
        
        self.RSI = cl.Buffer(ctx, mf.READ_ONLY, size=rsi.nbytes)
        self.GSI = cl.Buffer(ctx, mf.READ_ONLY, size=rsi.nbytes)
        self.BSI = cl.Buffer(ctx, mf.READ_ONLY, size=rsi.nbytes)
        self.STR = {}; self.STG = {}; self.STB = {}; self.skyTexnSize = {}
        self.US = cl.Buffer(ctx, mf.READ_ONLY, size=us.nbytes)
        self.VS = cl.Buffer(ctx, mf.READ_ONLY, size=us.nbytes)
        
        self.RO = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes)
        self.GO = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes)
        self.BO = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes)
        
        self.r2 = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes//4)
        self.g2 = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes//4)
        self.b2 = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes//4)

        self.r3 = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes//4)
        self.g3 = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes//4)
        self.b3 = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes//4)

        self.PC = makeRBuf(np.zeros((max_particles, 4), dtype="float32").nbytes)
        self.PO = makeRBuf(np.zeros((max_particles, 4), dtype="uint16").nbytes)
        self.PT = {}; self.PTsize = {}

        self.XYZ = []
        self.UV = []
        self.VN = []
        self.LI = []
        self.SP = []
        self.ZZ = []
        self.gSize = []

        self.BN = {}
        self.oldBT = {}
        self.BT = {}
        self.oldtt = {}

        self.TI = []
        self.TN = []
        self.TO = []
        
        self.TR = []
        self.TG = []
        self.TB = []
        self.TA = []
        self.texSize = []

        self.SHADOWMAP = {}
        
        self.RRR = {}
        self.GRR = {}
        self.BRR = {}
        self.reflTexSize = {}

        p = np.ones((3,), dtype="float32")
        self.VIEWPOS = cl.Buffer(ctx, mf.READ_ONLY, size=p.nbytes)
        self.VIEWMAT = cl.Buffer(ctx, mf.READ_ONLY, size=4*p.nbytes)

        self.LT = max_uv
        mn = np.ones((self.LT,), dtype="int32")
        self.AL = cl.Buffer(ctx, mf.READ_WRITE, size=mn.nbytes)

        self.hro = np.ones((h, w), dtype="uint16")
        self.hgo = np.ones((h, w), dtype="uint16")
        self.hbo = np.ones((h, w), dtype="uint16")
        self.hdb = np.ones((h, w), dtype="float32")

        self.useCompound = []

    def setScaleCull(self, s, cx, cy):
        self.sScale = np.float32(s)
        self.caX, self.caY = np.float32(cx), np.float32(cy)

    def setPSTex(self, rgb, name):
        rr = np.array(rgb[:,0]).astype("uint16")#rr = align34(rgb.astype("uint16"))
        self.PT[name] = makeRBuf(rr.nbytes)
        cl.enqueue_copy(cq, self.PT[name], rr, is_blocking=False)
        self.PTsize[name] = np.int32(np.sqrt(rr.shape[0]))

    def drawPSClouds(self, si, xyz, opacity, size, name, pos, randPos, skyHemi):
        vs = np.int32(xyz.shape[0]//BLOCK_SIZE + 1)
        cl.enqueue_copy(cq, self.PC, align34(xyz.astype("float32")))
        sm = self.SHADOWMAP[si]
        cloud.ps(cq, (vs, 1), (BLOCK_SIZE, 1),
                 self.RO, self.GO, self.BO, self.DB,
                 self.PC, *pos.astype("float32"), *randPos.astype("float32"),
                 np.float32(opacity), np.int32(size),
                 self.PT[name], self.PTsize[name],
                 self.LInt, *np.array(skyHemi, dtype="float32"),
                 sm["map"], sm["dim2"], sm["scale"],
                 sm["vec"], sm["pos"],
                 self.VIEWPOS, self.VIEWMAT,
                 self.sScale, self.W, self.H,
                 self.caX, self.caY, np.int32(xyz.shape[0]),
                 g_times_l=True)
    
    def drawPSTex(self, xyz, color, opacity, size, name):
        vs = np.int32(xyz.shape[0]//BLOCK_SIZE + 1)
        cl.enqueue_copy(cq, self.PC, align34(xyz.astype("float32")))
        cl.enqueue_copy(cq, self.PO, align34(color.astype("uint16")))
        particlestex.ps(cq, (vs, 1), (BLOCK_SIZE, 1),
                     self.RO, self.GO, self.BO, self.DB,
                     self.PC, self.PO,
                     np.float32(opacity), np.int32(size),
                     self.PT[name], self.PTsize[name],
                     self.VIEWPOS, self.VIEWMAT,
                     self.sScale, self.W, self.H,
                     self.caX, self.caY, np.int32(xyz.shape[0]),
                     g_times_l=True)

    def drawPS(self, xyz, color, opacity, size):
        vs = np.int32(xyz.shape[0]//BLOCK_SIZE + 1)
        cl.enqueue_copy(cq, self.PC, align34(xyz.astype("float32")))
        cl.enqueue_copy(cq, self.PO, align34(color.astype("uint16")))
        particles.ps(cq, (vs, 1), (BLOCK_SIZE, 1),
                     self.RO, self.GO, self.BO, self.DB,
                     self.PC, self.PO,
                     np.float32(opacity), np.int32(size),
                     self.VIEWPOS, self.VIEWMAT,
                     self.sScale, self.W, self.H,
                     self.caX, self.caY, np.int32(xyz.shape[0]),
                     g_times_l=True)

    def addBoneWeights(self, tn, bw):
        bb = bw.astype("int8")
        self.BN[tn] = makeRBuf(bb.nbytes)
        cl.enqueue_copy(cq, self.BN[tn], bb)

    def initBoneTransforms(self, name, bn):
        s = np.zeros((4,4),dtype="float32")
        self.BT[name] = makeRBuf(bn*s.nbytes)
        self.oldBT[name] = makeRBuf(bn*s.nbytes)
        self.oldtt[name] = np.tile(np.identity(4), (bn,1)).astype("float32")

    def initBoneOrigin(self, o, bn, tn):
        o = o.astype("float32")
        vs = np.int32(self.gSize[tn]//BLOCK_SIZE + 1)
        skel.offset(cq, (vs, 1), (BLOCK_SIZE, 1),
                       self.XYZ[tn], self.BN[tn],
                       np.int8(bn), o[0], o[1], o[2],
                       self.gSize[tn],
                       g_times_l=True)

    def setBoneTransform(self, name, bt):
        tt = self.invbt(bt, 1).astype("float32")
        cl.enqueue_copy(cq, self.BT[name], tt)
        cl.enqueue_copy(cq, self.oldBT[name], self.oldtt[name])
        self.oldtt[name] = self.invbt(tt)
        
    def boneTransform(self, cStart, cEnd, tn, name, offset=0):
        vs = np.int32((cEnd - cStart)//BLOCK_SIZE + 1)
        skel.transform(cq, (vs, 1), (BLOCK_SIZE, 1),
                       self.XYZ[tn], self.VN[tn],
                       self.BN[tn], self.oldBT[name], self.BT[name],
                       np.int32(cStart), np.int32(cEnd), np.int8(offset),
                       g_times_l=True)

    def invbt(self, b, tt=-1):
        for i in range(b.shape[0]//4):
            b[4*i:4*i+3,:3] = np.transpose(b[4*i:4*i+3,:3])
            b[4*i+3] *= tt
        return b

    def highlight(self, b, o, sr, lc, hc, hl, commit, tn, showAll=1):
        o = o.astype("float32")
        vs = np.int32(self.gSize[tn]//BLOCK_SIZE + 1)
        sct.highlight(cq, (vs, 1), (BLOCK_SIZE, 1),
                   self.XYZ[tn], self.BN[tn],
                   self.LI[tn], np.float32(lc), np.float32(hc), np.float32(hl),
                   np.int8(b), o[0], o[1], o[2], np.float32(sr),
                   np.int8(commit), np.int8(showAll), self.gSize[tn],
                   g_times_l=True)
        
    def getBoneWeights(self, tn):
        b = np.zeros((self.gSize[tn]//3,3), dtype="int8")
        cl.enqueue_copy(cq, b, self.BN[tn])
        return b
    
    def getVertPoints(self, tn):
        b = np.zeros((self.gSize[tn]//3,3,4), dtype="float32")
        cl.enqueue_copy(cq, b, self.XYZ[tn])
        return b
    def getVertNorms(self, tn):
        b = np.zeros((self.gSize[tn]//3,3,4), dtype="float32")
        cl.enqueue_copy(cq, b, self.VN[tn])
        return b

    def addTexture(self, r, g, b, mip=False):
        rr = r.astype("uint16")
        gg = g.astype("uint16")
        bb = b.astype("uint16")
        
        self.TR.append(makeRBuf(rr.nbytes))
        self.TG.append(makeRBuf(rr.nbytes))
        self.TB.append(makeRBuf(rr.nbytes))
        cl.enqueue_copy(cq, self.TR[-1], rr, is_blocking=False)
        cl.enqueue_copy(cq, self.TG[-1], gg, is_blocking=False)
        cl.enqueue_copy(cq, self.TB[-1], bb, is_blocking=False)
        
        if mip: self.texSize.append(np.int32(np.log2(mip)))
        else: self.texSize.append(np.int32(rr.shape[0]))
        return len(self.TR) - 1

    def addTextureGroup(self, xyz, uv, vn, r, g, b, mip=False):
        rr = r.astype("uint16")
        gg = g.astype("uint16")
        bb = b.astype("uint16")
        
        p = xyz.astype("float32")
        p = align34(p)
        
        n = vn.astype("float32")
        n = align34(n)
        
        uv = uv.astype("float32")
        
        self.XYZ.append(makeRBuf(p.nbytes))
        self.UV.append(makeRBuf(uv.nbytes))
        self.VN.append(makeRBuf(p.nbytes))
        self.LI.append(makeRBuf(uv.nbytes*2))

        self.SP.append(makeRBuf(uv.nbytes))
        self.ZZ.append(makeRBuf(uv.nbytes//2))
        self.TI.append(makeRBuf(uv.nbytes//6))
        gs = int(p.shape[0] / 3 / BLOCK_SIZE)+1
        ib = np.ones((1,),dtype="int32").nbytes
        self.TN.append(makeRBuf(gs*ib))
        self.TO.append(makeRBuf(uv.nbytes//6))
        
        self.TR.append(makeRBuf(rr.nbytes))
        self.TG.append(makeRBuf(rr.nbytes))
        self.TB.append(makeRBuf(rr.nbytes))
        
        cl.enqueue_copy(cq, self.XYZ[-1], p, is_blocking=False)
        cl.enqueue_copy(cq, self.UV[-1], uv, is_blocking=False)
        cl.enqueue_copy(cq, self.VN[-1], n, is_blocking=False)
        
        cl.enqueue_copy(cq, self.TR[-1], rr, is_blocking=False)
        cl.enqueue_copy(cq, self.TG[-1], gg, is_blocking=False)
        cl.enqueue_copy(cq, self.TB[-1], bb, is_blocking=False)
        if mip:
            self.texSize.append(np.int32(np.log2(mip)))
        else:
            self.texSize.append(np.int32(rr.shape[0]))
        self.gSize.append(np.int32(p.shape[0]))
        self.useCompound.append(False)
        
        return len(self.TR)-1

    def copyTo(self, tn, vp, vn, cstart=0):
        cl.enqueue_copy(cq, self.XYZ[tn], align34(vp.astype("float32")),
                        device_offset=cstart*3*4*4, is_blocking=False)
        cl.enqueue_copy(cq, self.VN[tn], align34(vn.astype("float32")),
                        device_offset=cstart*3*4*4)

    def addTexAlpha(self, a):
        self.TA.append(cl.Buffer(ctx, mf.READ_ONLY, size=a.nbytes))
        cl.enqueue_copy(cq, self.TA[-1], a, is_blocking=False)

    def setReflTex(self, name, r, g, b, size):
        rr = r.astype("uint16")
        self.RRR[name] = makeRBuf(rr.nbytes)
        self.GRR[name] = makeRBuf(rr.nbytes)
        self.BRR[name] = makeRBuf(rr.nbytes)
        cl.enqueue_copy(cq, self.RRR[name], rr, is_blocking=False)
        cl.enqueue_copy(cq, self.GRR[name], g.astype("uint16"), is_blocking=False)
        cl.enqueue_copy(cq, self.BRR[name], b.astype("uint16"), is_blocking=False)
        self.reflTexSize[name] = np.int32(size/2)
        return len(self.reflTexSize) - 1
    
    def setSkyTex(self, r, g, b, size):
        cl.enqueue_copy(cq, self.RSI, r.astype("uint16"), is_blocking=False)
        cl.enqueue_copy(cq, self.GSI, g.astype("uint16"), is_blocking=False)
        cl.enqueue_copy(cq, self.BSI, b.astype("uint16"), is_blocking=False)
        self.skyTexSize = np.int32(size)

    def setPos(self, vc):
        cl.enqueue_copy(cq, self.VIEWPOS, vc.astype("float32"))
    def setVM(self, vM):
        v = align34(vM)
        cl.enqueue_copy(cq, self.VIEWMAT, v.astype("float32"))

    def transform(self, oldRM, rotMat, origin, cStart, cEnd, tn):
        oo = origin.astype("float32")
        o = makeRBuf(oo.nbytes)
        cl.enqueue_copy(cq, o, oo)
        rM = align34(rotMat.astype("float32").T)
        oRM = align34(oldRM.astype("float32").T)
        r = makeRBuf(rM.nbytes)
        rr = makeRBuf(oRM.nbytes)
        cl.enqueue_copy(cq, r, rM)
        cl.enqueue_copy(cq, rr, oRM)
        vs = np.int32((cEnd - cStart)//BLOCK_SIZE + 1)
        vert.transform(cq, (vs, 1), (BLOCK_SIZE, 1),
                       self.XYZ[tn], self.VN[tn],
                       rr, r, o,
                       np.int32(cStart), np.int32(cEnd),
                       g_times_l=True)

    def setupWave(self, origin, wDir, wLen, wAmp, wSpd, numW):
        self.WAVEO = [np.float32(x) for x in origin]
        self.WAVED = [np.float32(x) for x in wDir.reshape((-1,))]
        
        wl = wLen.astype("float32")
        self.WAVELEN = makeRBuf(wl.nbytes)
        cl.enqueue_copy(cq, self.WAVELEN, wl)
        wa = wAmp.astype("float32")
        self.WAVEAMP = makeRBuf(wa.nbytes)
        cl.enqueue_copy(cq, self.WAVEAMP, wa)
        ws = wSpd.astype("float32")
        self.WAVESPD = makeRBuf(ws.nbytes)
        cl.enqueue_copy(cq, self.WAVESPD, ws)
        #self.WNUM = np.int8(ws.shape[0])
        self.WNUM = [np.int8(numW), np.int8(ws.shape[0] - numW)]

    def updateWave(self, pScale, stTime, tn):
        vs = np.int32(self.gSize[tn] // BLOCK_SIZE + 1)
        wave.wave(cq, (vs, 1), (BLOCK_SIZE, 1),
                  self.XYZ[tn], self.VN[tn],
                  *self.WAVEO, *self.WAVED, np.float32(pScale),
                  np.float32(time.time() - stTime),
                  self.WAVELEN, self.WAVEAMP,
                  self.WAVESPD, *self.WNUM, self.gSize[tn],
                  g_times_l=True)
        
    def vertLight(self, mask, dirI, dirD, pointI=None, pointP=None,
                  spotI=None, spotD=None, spotP=None):
        i = dirI.astype("float32")
        i = align34(i)
        d = dirD.astype("float32")
        d = align34(d)
        self.dirInt = makeRBuf(i.nbytes)
        self.dirDir = makeRBuf(d.nbytes)
        cl.enqueue_copy(cq, self.dirInt, i)
        cl.enqueue_copy(cq, self.dirDir, d)
        self.numDirs = np.int8(dirI.shape[0])

        if pointI is None:
            pass
        elif pointI is 1:
            self.pointInt = makeRBuf(4)
            self.pointPos = makeRBuf(4)
            self.numPoints = np.int16(0)
        else:
            i = pointI.astype("float32")
            i = align34(i)
            d = pointP.astype("float32")
            d = align34(d)
            self.pointInt = makeRBuf(i.nbytes)
            self.pointPos = makeRBuf(d.nbytes)
            cl.enqueue_copy(cq, self.pointInt, i)
            cl.enqueue_copy(cq, self.pointPos, d)
            self.numPoints = np.int16(pointI.shape[0])
        if spotI is None:
            pass
        elif spotI is 1:
            self.spotInt = makeRBuf(4)
            self.spotDir = makeRBuf(4)
            self.spotPos = makeRBuf(4)
            self.numSpots = np.int32(0)
        else:
            i = spotI.astype("float32")
            i = align34(i)
            d = spotD.astype("float32")
            d = align34(d)
            p = spotP.astype("float32")
            p = align34(p)
            self.spotInt = makeRBuf(i.nbytes)
            self.spotDir = makeRBuf(d.nbytes)
            self.spotPos = makeRBuf(p.nbytes)
            cl.enqueue_copy(cq, self.spotInt, i)
            cl.enqueue_copy(cq, self.spotDir, d)
            cl.enqueue_copy(cq, self.spotPos, p)
            self.numSpots = np.int32(spotI.shape[0])
        
        for tn in range(len(self.gSize)):
            if mask[tn]:
                vs = np.int32(self.gSize[tn]//BLOCK_SIZE + 1)
                vert.vertL(cq, (vs, 1), (BLOCK_SIZE, 1),
                           self.XYZ[tn], self.VN[tn], self.LI[tn],
                           self.dirInt, self.dirDir,
                           self.pointInt, self.pointPos,
                           self.spotInt, self.spotDir, self.spotPos,
                           self.ambLight,
                           self.numDirs, self.numPoints, self.numSpots,
                           self.gSize[tn], g_times_l=True)

    def setHostSkyTex(self, tex):
        self.hostSTex = tex.astype("uint16")
        self.stSize = tex.shape[0]
        
    def drawCubeMap(self, size, pos, *shaders, maskNum=None):
        out = []
        vv = np.array([[0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0],
                       [0, 1, 0], [0, -1, 0]])
        pi = 3.1415926535
        p = [[0, 0], [pi/2, 0], [pi, 0], [3/2*pi, 0],
             [pi/2, -pi/2], [pi/2, pi/2]]
        self.setPos(pos)
        tempcx = self.caX
        tempcy = self.caY
        tempsc = self.sScale
        self.caX = np.float32(1.5)
        self.caY = np.float32(1.5)
        self.sScale = np.float32(size/2)
        tempW = self.W
        tempH = self.H
        self.W = np.int32(size)
        self.H = np.int32(size)

        tempRGBZ = (self.RO, self.GO, self.BO, self.DB)
        fb = np.array(self.hostSTex[:,:self.stSize])
        self.RO = makeRBuf(fb.nbytes//3)
        self.GO = makeRBuf(fb.nbytes//3)
        self.BO = makeRBuf(fb.nbytes//3)
        self.DB = makeRBuf(fb.nbytes//3*2)

        if maskNum is None:
            rMask = None
        elif type(maskNum) is int:
            rMask = list([False]*len(shaders[0]))
            rMask[maskNum] = True
        else:
            rMask = list([False]*len(shaders[0]))
            for i in maskNum:
                rMask[i] = True
        
        for i in range(6):
            fb = np.array(self.hostSTex[:,i*self.stSize:(i+1)*self.stSize])
            self.setVM(viewMat(*p[i]))
            self.clearZBuffer()
            cl.enqueue_copy(cq, self.RO, np.array(fb[:,:,0]))
            cl.enqueue_copy(cq, self.GO, np.array(fb[:,:,1]))
            cl.enqueue_copy(cq, self.BO, np.array(fb[:,:,2]))
            self.drawAll(*shaders, mask=rMask)
            hr = np.zeros((size,size), dtype="uint16")
            hg = np.zeros((size,size), dtype="uint16")
            hb = np.zeros((size,size), dtype="uint16")
            cl.enqueue_copy(cq, hr, self.RO)
            cl.enqueue_copy(cq, hg, self.GO)
            cl.enqueue_copy(cq, hb, self.BO)
            out.append(np.stack((hr,hg,hb),axis=2))

        self.W = tempW
        self.H = tempH
        self.caX = tempcx
        self.caY = tempcy
        self.sScale = tempsc
        self.RO, self.GO, self.BO, self.DB = tempRGBZ

        return np.stack(out, axis=0)
    
    def drawAll(self, useAlpha, useShadow, useMip, useRefl,
                mask=None, shadowIds=[0,1], ortho=False, useOpacitySM=False):
        if mask is None:
            mask = [False] * len(useAlpha)
        gsn = []
        #a = time.time()

        if ortho:
            for tn in range(len(self.gSize)):
                if mask[tn]:
                    gsn.append(0)
                    continue
                gs = np.int32(int(self.gSize[tn] / 3 / BLOCK_SIZE)+1)
                gsn.append(gs)
                trisetupOrtho.setup(cq, (gs, 1), (BLOCK_SIZE, 1),
                               self.XYZ[tn], self.TI[tn], self.TN[tn],
                               self.SP[tn], self.ZZ[tn],
                               self.VIEWPOS, self.VIEWMAT,
                               np.float32(0),
                               self.sScale, self.W, self.H,
                               self.caX, self.caY, np.int32(self.gSize[tn]//3),
                               g_times_l=True)
        else:
            for tn in range(len(self.gSize)):
                if mask[tn]:
                    gsn.append(0)
                    continue
                gs = np.int32(int(self.gSize[tn] / 3 / BLOCK_SIZE)+1)
                gsn.append(gs)
                trisetup.setup(cq, (gs, 1), (BLOCK_SIZE, 1),
                               self.XYZ[tn], self.VN[tn],
                               self.TI[tn], self.TN[tn],
                               self.SP[tn], self.ZZ[tn],
                               self.VIEWPOS, self.VIEWMAT,
                               self.sScale, self.W, self.H,
                               self.caX, self.caY, np.int32(self.gSize[tn]//3),
                               g_times_l=True)

        #print("Trisetup:", time.time() - a)
        #a = time.time()
        for tn in range(len(self.gSize)):
            if mask[tn]: continue
            gs = gsn[tn]
            gather.setup(cq, (gs, 1), (BLOCK_SIZE, 1),
                         self.TI[tn], self.TN[tn],
                         self.TO[tn], self.AL, np.int32(tn),
                         gs, g_times_l=True)

        #print("Gather:", time.time()-a)
        #a = time.time()

        newSize = np.zeros((self.LT,), dtype="int32")
        cl.enqueue_copy(cq, newSize, self.AL, is_blocking=True)
        nsn = newSize // BLOCK_SIZE + 1
        
        #print("After:", time.time()-a)
        #a = time.time()

        sm = self.SHADOWMAP[shadowIds[0]]
        sm1 = self.SHADOWMAP[shadowIds[1]]
        for tn in range(len(self.gSize)):
            if mask[tn]: continue
            ns = nsn[tn]
            if newSize[tn] > 0:
                sm = self.SHADOWMAP[shadowIds[0]]
                sm1 = self.SHADOWMAP[shadowIds[1]]
                if useAlpha[tn]:
                    sm = self.SHADOWMAP[0]
                    sm1 = self.SHADOWMAP[1]
                    drawA.drawTex(cq, (ns, 1), (BLOCK_SIZE, 1),
                              self.TO[tn],
                             self.RO, self.GO, self.BO,
                             self.DB, self.SP[tn], self.ZZ[tn],
                             self.UV[tn], self.LI[tn], self.VN[tn], self.XYZ[tn],
                             self.LInt, self.LDir,
                             sm["map"], sm["dim2"], sm["scale"],
                             sm["vec"], sm["pos"],
                             sm1["map"], sm1["dim2"], sm1["scale"],
                             sm1["vec"], sm1["pos"],
                             self.ambLight,
                             self.TR[tn], self.TG[tn], self.TB[tn],
                             self.TA[useAlpha[tn] - 1],
                             np.int8(useShadow[tn]),
                             self.W, self.H, np.int32(newSize[tn]),
                             self.texSize[tn],
                             g_times_l=True)
                elif (useMip[tn] and useOpacitySM):
                    drawTriSOp.drawTex(cq, (ns, 1), (BLOCK_SIZE, 1),
                              self.TO[tn],
                             self.RO, self.GO, self.BO,
                             self.DB, self.SP[tn], self.ZZ[tn],
                             self.UV[tn], self.LI[tn],
                             self.VN[tn], self.XYZ[tn],
                             self.LInt, self.LDir,
                             sm["map"], sm["dim2"], sm["scale"],
                             sm["vec"], sm["pos"],
                             sm1["opmap"], sm1["dim2"], sm1["scale"],
                             sm1["vec"], sm1["pos"],
                             self.ambLight,
                             self.TR[tn], self.TG[tn], self.TB[tn],
                             self.W, self.H, np.int32(newSize[tn]),
                             self.texSize[tn],
                             g_times_l=True)
                elif useMip[tn]:
                    if tn == 0: b = 4
                    else: b = 0.2
                    drawTriS.drawTex(cq, (ns, 1), (BLOCK_SIZE, 1),
                              self.TO[tn],
                             self.RO, self.GO, self.BO,
                             self.DB, self.SP[tn], self.ZZ[tn],
                             self.UV[tn], self.LI[tn],
                             self.VN[tn], self.XYZ[tn],
                             self.LInt, self.LDir,
                             sm["map"], sm["dim2"], sm["scale"],
                             sm["vec"], sm["pos"],
                             sm1["map"], sm1["dim2"], sm1["scale"],
                             sm1["vec"], sm1["pos"],
                             self.ambLight, np.float32(b),
                             self.TR[tn], self.TG[tn], self.TB[tn],
                             self.W, self.H, np.int32(newSize[tn]),
                             self.texSize[tn],
                             g_times_l=True)
                elif (useRefl[tn] is not None) and (useRefl[tn] is not False):
                    refName = useRefl[tn]
                    drawCMT.drawTex(cq, (ns, 1), (BLOCK_SIZE, 1),
                              self.TO[tn],
                             self.RO, self.GO, self.BO,
                             self.DB, self.SP[tn], self.ZZ[tn],
                             self.UV[tn], self.LI[tn],
                             self.VN[tn], self.XYZ[tn],
                             self.LInt, self.LDir,
                             sm["map"], sm["dim2"], sm["scale"],
                             sm["vec"], sm["pos"],
                             self.ambLight,
                             self.TR[tn], self.TG[tn], self.TB[tn],
                             np.int8(1),
                             self.VIEWPOS,
                             self.RRR[refName], self.GRR[refName], self.BRR[refName],
                             np.float32(1.33),
                             self.W, self.H, np.int32(newSize[tn]),
                             self.texSize[tn], self.reflTexSize[refName],
                             g_times_l=True)
                else:
                    draw.drawTex(cq, (ns, 1), (BLOCK_SIZE, 1),
                              self.TO[tn],
                             self.RO, self.GO, self.BO,
                             self.DB, self.SP[tn], self.ZZ[tn],
                             self.UV[tn], self.LI[tn], self.VN[tn], self.XYZ[tn],
                             self.LInt, self.LDir,
                             sm["map"], sm["dim2"], sm["scale"],
                             sm["vec"], sm["pos"],
                             self.ambLight,
                             self.TR[tn], self.TG[tn], self.TB[tn],
                             np.int8(useShadow[tn]),
                             self.W, self.H, np.int32(newSize[tn]),
                             self.texSize[tn],
                             g_times_l=True)

        #print("Pixel:", time.time()-a)
        #a = time.time()
    
    def clearZBuffer(self):
        s = 2; t = 2
        clearzb.clearFrame(cq, (s, s), (t, t), self.DB, self.W, self.H,
                           np.int32(t), np.int32(s*t), g_times_l=True)
    def gamma(self):
        s = 4; t = 4
        gamma.g(cq, (s, s), (t, t), self.RO, self.GO, self.BO,
                self.W, self.H, np.int32(t), np.int32(s*t),
                np.int32(np.ceil(self.H/(s*t))), g_times_l=True)

    def blur(self):
        s = 4; t = 4
        e = blur1.blurH(cq, (s, s), (t, t), self.RO, self.GO, self.BO,
                    self.r2, self.g2, self.b2,
                    self.r3, self.g3, self.b3,
                    self.W, self.H, np.int32(t), np.int32(s*t),
                    np.float32(np.ceil(self.H/(s*t))), g_times_l=True)
        blur2.blurV(cq, (s, s), (t, t), self.r2, self.g2, self.b2,
                    self.r3, self.g3, self.b3,
                    self.RO, self.GO, self.BO,
                    self.W, self.H, np.int32(t), np.int32(s*t),
                    np.float32(np.ceil(self.H/(s*t))), g_times_l=True,
                    wait_for=[e])
    
    def skydraw(self, sp, u, v):
        gs = int(len(sp) / 3 / BLOCK_SIZE)+1
        self.test = sp
        cl.enqueue_copy(cq, self.SPS, sp.astype("int32"), is_blocking=False)
        cl.enqueue_copy(cq, self.US, u.astype("float32"), is_blocking=False)
        cl.enqueue_copy(cq, self.VS, v.astype("float32"))

        prg2.drawSky(cq, (gs, 1), (BLOCK_SIZE, 1),
                     self.RO, self.GO, self.BO,
                     self.SPS, self.US, self.VS,
                     self.RSI, self.GSI, self.BSI,
                     self.W, self.H,
                     np.int32(u.shape[0]//3), self.skyTexSize,
                     g_times_l=True)
        return True
    
    def skydraw1(self, sp, u, v, tn, tr=None, tg=None, tb=None):
        gs = int(len(sp) / 3 / BLOCK_SIZE)+1
        self.test = sp
        cl.enqueue_copy(cq, self.SPS, sp.astype("int32"), is_blocking=False)
        cl.enqueue_copy(cq, self.US, u.astype("float32"), is_blocking=False)
        cl.enqueue_copy(cq, self.VS, v.astype("float32"))

        if tn not in self.STR:
            sz = np.zeros_like(tr, dtype="uint16").nbytes
            self.STR[tn] = makeRBuf(sz)
            self.STG[tn] = makeRBuf(sz)
            self.STB[tn] = makeRBuf(sz)
            cl.enqueue_copy(cq, self.STR[tn], tr.astype("uint16"))
            cl.enqueue_copy(cq, self.STG[tn], tg.astype("uint16"))
            cl.enqueue_copy(cq, self.STB[tn], tb.astype("uint16"))
            self.skyTexnSize[tn] = np.int32(tr.shape[0])

        sun.drawSky(cq, (gs, 1), (BLOCK_SIZE, 1),
                     self.RO, self.GO, self.BO,
                     self.SPS, self.US, self.VS,
                     self.STR[tn], self.STG[tn], self.STB[tn],
                     self.LInt,
                     self.W, self.H,
                     np.int32(u.shape[0]//3), self.skyTexnSize[tn],
                     g_times_l=True)
        return True
    
    def clearShadowMap(self, i):
        sm = self.SHADOWMAP[i]
        s = 4; t = 4
        clearzb.clearFrame(cq, (s, s), (t, t), sm["map"],
                           sm["dim"], sm["dim"],
                           np.int32(t), np.int32(s*t), g_times_l=True)
        
    def clearShadowOpMap(self, i):
        s = self.SHADOWMAP[i]
        a = np.full((s["dim"], s["dim"]), 0, dtype="uint16")
        cl.enqueue_copy(cq, s["opmap"], a)

    def addShadowMap(self, i, size, scale, ambLight=None, useGI=False):
        a = np.full((size, size), 256*256-1, dtype="float32")
        b = np.ones((3, 4), dtype="float32")
        s = {}
        s["map"] = cl.Buffer(ctx, mf.READ_WRITE, size=a.nbytes)
        s["vec"] = cl.Buffer(ctx, mf.READ_ONLY, size=b.nbytes)

        p = np.ones((3,), dtype="float32")
        s["pos"] = cl.Buffer(ctx, mf.READ_ONLY, size=p.nbytes)
        
        s["dim"] = np.int32(size)
        s["dim2"] = np.int32(size/2)
        s["scale"] = np.float32(scale)

        if ambLight is not None:
            self.ambLight = np.float32(ambLight)

        if useGI:
            s["normbuf"] = cl.Buffer(ctx, mf.WRITE_ONLY, size=a.nbytes*4)
            s["Ro"] = cl.Buffer(ctx, mf.WRITE_ONLY, size=a.nbytes//2)
            s["Go"] = cl.Buffer(ctx, mf.WRITE_ONLY, size=a.nbytes//2)
            s["Bo"] = cl.Buffer(ctx, mf.WRITE_ONLY, size=a.nbytes//2)

        self.SHADOWMAP[i] = s

    def addOpacityMap(self, i):
        s = self.SHADOWMAP[i]
        a = np.full((s["dim"], s["dim"]), 0, dtype="uint16")
        s["opmap"] = cl.Buffer(ctx, mf.READ_WRITE, size=a.nbytes)

    def placeShadowMap(self, i, pos, facing, ambLight=None):
        sm = self.SHADOWMAP[i]
        p = np.array(pos).astype("float32")
        cl.enqueue_copy(cq, sm["pos"], p)
        f = viewMat(*facing)
        f = align34(f)
        cl.enqueue_copy(cq, sm["vec"], f.astype("float32"))
        sm["vecnp"] = f
        if ambLight is not None:
            self.ambLight = np.float32(ambLight)
    
    def shadowMap(self, i, whichCast, useAlpha, bias):
        shBias = np.float32(bias)
        tsn = []
        gsn = []
        sm = self.SHADOWMAP[i]
        for tn in range(len(self.gSize)):
            if whichCast[tn]:
                tsn.append(tn)
                gs = np.int32(int(self.gSize[tn] / 3 / BLOCK_SIZE)+1)
                gsn.append(gs)
                trisetupOrtho.setup(cq, (gs, 1), (BLOCK_SIZE, 1),
                               self.XYZ[tn], self.TI[tn], self.TN[tn],
                               self.SP[tn], self.ZZ[tn],
                               sm["pos"], sm["vec"],
                               shBias,
                               sm["scale"],
                               sm["dim"], sm["dim"],
                               self.caX, self.caY, np.int32(self.gSize[tn]//3),
                               g_times_l=True)

        for i in range(len(tsn)):
            gs = gsn[i]
            tn = tsn[i]
            gather.setup(cq, (gs, 1), (BLOCK_SIZE, 1),
                         self.TI[tn], self.TN[tn],
                         self.TO[tn], self.AL, np.int32(tn),
                         gs, g_times_l=True)
        
        newSize = np.zeros((self.LT,), dtype="int32")
        cl.enqueue_copy(cq, newSize, self.AL, is_blocking=True)
        nsn = newSize // BLOCK_SIZE + 1

        #print(nsn, "\n", newSize)
        for i in range(len(tsn)):
            tn = tsn[i]
            ns = nsn[tn]
            #print("ns:", ns, "tn:", tn, "newsize[tn]:", newSize[tn])
            if (newSize[tn] > 0):
                if not useAlpha[tn] or (tn in [14, 19]):
                    sh.draw(cq, (ns, 1), (BLOCK_SIZE, 1),
                            self.TO[tn],
                            sm["map"], self.SP[tn], self.ZZ[tn],
                            sm["dim"], sm["dim"], np.int32(newSize[tn]),
                            g_times_l=True)
                else:
                    shA.draw(cq, (ns, 1), (BLOCK_SIZE, 1),
                             self.TO[tn],
                             sm["map"], self.SP[tn], self.ZZ[tn],
                             self.UV[tn],
                             self.TA[useAlpha[tn] - 1], self.texSize[tn],
                             sm["dim"], sm["dim"], np.int32(newSize[tn]),
                             g_times_l=True)

            cq.flush()

    def shadowClouds(self, i, xyz, size, name, pos, dev):
        sm = self.SHADOWMAP[i]
        
        vs = np.int32(xyz.shape[0]//BLOCK_SIZE + 1)
        cl.enqueue_copy(cq, self.PC, align34(xyz.astype("float32")))
                
        cloudsh.sh(cq, (vs, 1), (BLOCK_SIZE, 1),
                   sm["map"], self.PC, np.int32(size),
                   self.PT[name], self.PTsize[name],
                   sm["pos"], sm["vec"],
                   *pos.astype("float32"), *dev.astype("float32"),
                   sm["scale"], sm["dim"], sm["dim"],
                   self.caX, self.caY, np.int32(xyz.shape[0]),
                   g_times_l=True)
    
    def shadowCloudsOpacity(self, i, xyz, size, name, pos, dev):
        sm = self.SHADOWMAP[i]
        
        vs = np.int32(xyz.shape[0]//BLOCK_SIZE + 1)
        cl.enqueue_copy(cq, self.PC, align34(xyz.astype("float32")))
                
        cloudshop.sh(cq, (vs, 1), (BLOCK_SIZE, 1),
                   sm["opmap"], self.PC, np.int32(size),
                   sm["pos"], sm["vec"],
                   *pos.astype("float32"), *dev.astype("float32"),
                   sm["scale"], sm["dim"], sm["dim"],
                   self.caX, self.caY, np.int32(xyz.shape[0]),
                   g_times_l=True)

    def setPrimaryLight(self, dirI, dirD):
        i = dirI.astype("float32")
        i = align34(i)
        d = dirD.astype("float32")
        d = align34(d)
        self.LInt = makeRBuf(i.nbytes)
        self.LDir = makeRBuf(d.nbytes)
        cl.enqueue_copy(cq, self.LInt, i)
        cl.enqueue_copy(cq, self.LDir, d)

    def drawDirectional(self, i, whichCast, shBias=np.float32(0.1)):
        tsn = []
        gsn = []
        sm = self.SHADOWMAP[i]
        for tn in range(len(self.gSize)):
            if whichCast[tn]:
                tsn.append(tn)
                gs = np.int32(int(self.gSize[tn] / 3 / BLOCK_SIZE)+1)
                gsn.append(gs)
                trisetupOrtho.setup(cq, (gs, 1), (BLOCK_SIZE, 1),
                               self.XYZ[tn], self.TI[tn], self.TN[tn],
                               self.SP[tn], self.ZZ[tn],
                               sm["pos"], sm["vec"],
                               shBias,
                               sm["scale"],
                               sm["dim"], sm["dim"],
                               self.caX, self.caY, np.int32(self.gSize[tn]//3),
                               g_times_l=True)

        for i in range(len(tsn)):
            gs = gsn[i]
            tn = tsn[i]
            gather.setup(cq, (gs, 1), (BLOCK_SIZE, 1),
                         self.TI[tn], self.TN[tn],
                         self.TO[tn], self.AL, np.int32(tn),
                         gs, g_times_l=True)
        
        newSize = np.zeros((self.LT,), dtype="int32")
        cl.enqueue_copy(cq, newSize, self.AL, is_blocking=True)
        nsn = newSize // BLOCK_SIZE + 1

        for i in range(len(tsn)):
            tn = tsn[i]
            ns = nsn[tn]
            if newSize[tn] > 0:
                if True: #if not useAlpha[tn]:
                    drawNorm.draw(cq, (ns, 1), (BLOCK_SIZE, 1),
                             self.TO[tn],
                             sm["Ro"], sm["Go"], sm["Bo"],
                             sm["normbuf"],
                             sm["map"], self.SP[tn], self.ZZ[tn],
                             self.UV[tn], self.VN[tn],
                             self.TR[tn], self.TG[tn], self.TB[tn],
                             sm["dim"], sm["dim"], np.int32(newSize[tn]),
                             self.texSize[tn], g_times_l=True)
##                else:
##                    shA.draw(cq, (ns, 1), (BLOCK_SIZE, 1),
##                             self.TO[tn],
##                             sm["map"], self.SP[tn], self.ZZ[tn],
##                             self.UV[tn],
##                             self.TA[useAlpha[tn] - 1], self.texSize[tn],
##                             sm["dim"], sm["dim"], np.int32(newSize[i]),
##                             g_times_l=True)

    def getGIM(self, i):
        sm = self.SHADOWMAP[i]
        size = sm["dim"]
        hr = np.zeros((size,size), dtype="uint16")
        hg = np.zeros((size,size), dtype="uint16")
        hb = np.zeros((size,size), dtype="uint16")
        hz = np.zeros((size,size), dtype="float32")
        hn = np.zeros((size,size,4), dtype="float32")
        cq.flush()
        cl.enqueue_copy(cq, hr, sm["Ro"])
        cl.enqueue_copy(cq, hg, sm["Go"])
        cl.enqueue_copy(cq, hb, sm["Bo"])
        cl.enqueue_copy(cq, hz, sm["map"])
        cl.enqueue_copy(cq, hn, sm["normbuf"])
        
        out = [np.stack((hr,hg,hb),axis=2), hz, hn]
        return out
        
    def getSHM(self, i):
        sm = self.SHADOWMAP[i]
        s = np.empty((sm["dim"], sm["dim"]), dtype="float32")
        cl.enqueue_copy(cq, s, sm["map"])
        return s

    def getFrame(self):
        cl.enqueue_copy(cq, self.hro, self.RO, is_blocking=False)
        cl.enqueue_copy(cq, self.hgo, self.GO, is_blocking=False)
        cl.enqueue_copy(cq, self.hbo, self.BO, is_blocking=False)
        cl.enqueue_copy(cq, self.hdb, self.DB)
        
        return (self.hro, self.hgo, self.hbo, self.hdb)
