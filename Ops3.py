# OpenCL rasterization
# Multiple rigs

import numpy as np
import time
from Utils import viewMat

import pyopencl as cl

infotext = """Information about OpenCL devices on your computer.\n
If you are experiencing any problems or glitches running
AXI Visualizer, try changing devices. If only one device is
available try updating its drivers.
Open "Settings.txt" and change the CL option to 1:0 or whatever
number is in the square bracket [] of the device you want to use.\n
"""

pl = cl.get_platforms()
with open("CLInfo.txt", "w") as f:
    f.write(infotext)
    for i in range(len(pl)):
        p = pl[i]
        f.write("Platform: " + p.name + "\n")
        dv = p.get_devices()
        for j in range(len(dv)):
            f.write("  Device ["+str(i)+":"+str(j)+"] => " + dv[j].name + "\n")

import os
try:
    with open("Settings.txt") as f:
        for line in f:
            if line[:2] == "CL":
                os.environ["PYOPENCL_CTX"] = line[3:]
                break
except FileNotFoundError:
    with open("Settings.txt", "w") as f:
        f.write("Settings for AXI Visualizer\n\nCL=0:0\n")
    os.environ["PYOPENCL_CTX"] = "0:0"
except:
    os.environ["PYOPENCL_CTX"] = "0:0"

try:
    ctx = cl.create_some_context()
except cl.cffi_cl.RuntimeError:
    try:
        os.environ["PYOPENCL_CTX"] = os.environ["PYOPENCL_CTX"].split(":")[0]
        ctx = cl.create_some_context()
    except cl.cffi_cl.RuntimeError:
        raise

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
trisetup = makeProgram("trisetup.c", "Pipe/")
gather = makeProgram("gather.c", "Pipe/")

draw = makeProgram("drawtexcolsmshplerp.c")
drawA = makeProgram("drawtexcolsmshalphaplerp.c")
drawTriS = makeProgram("drawtexcolsmsh2plerpmiptri.c")
drawCMT = makeProgram("drawreflcubemapfresneltexsh.c")

wave = makeProgram("wave2.c", "VertShaders/")
skel = makeProgram("bone1.c", "VertShaders/")

sct = makeProgram("select.c", "VertShaders/")
swc = makeProgram("switch.c", "VertShaders/")

if PLATFORM == "intel":
    prg2 = makeProgram("drawskylerp_intel.c")
else:
    prg2 = makeProgram("drawskylerp.c")

trisetupOrtho = makeProgram("trisetupOrtho.c", "Pipe/")
sh = makeProgram("drawmin.c")
shA = makeProgram("drawminalpha.c")
clearzb = makeProgram("clearzb.c")

gamma = makeProgram("Post/gamma.c")

def makeRBuf(nbytes):
    return cl.Buffer(ctx, mf.READ_ONLY, size=nbytes)

def align34(a):
    return np.stack((a[:,0], a[:,1], a[:,2], np.zeros_like(a[:,0])), axis=1)

class CLDraw:
    def __init__(self, max_s, size_sky, max_uv, w, h):
        self.W = np.int32(w)
        self.H = np.int32(h)
        self.A = w*h

        sps = np.ones((max_s*3, 2), dtype="int32")
        us = np.zeros((max_s*3,), dtype="float32")
        rsi = np.ones((size_sky*size_sky*6,), dtype="uint16")
        
        ro = np.ones((h, w), dtype="uint16")
        db = np.full((h, w), 255, dtype="float32")
        
        self.SPS = cl.Buffer(ctx, mf.READ_ONLY, size=sps.nbytes)
        self.DB = cl.Buffer(ctx, mf.READ_WRITE, size=db.nbytes)
        
        self.RSI = cl.Buffer(ctx, mf.READ_ONLY, size=rsi.nbytes)
        self.GSI = cl.Buffer(ctx, mf.READ_ONLY, size=rsi.nbytes)
        self.BSI = cl.Buffer(ctx, mf.READ_ONLY, size=rsi.nbytes)
        self.US = cl.Buffer(ctx, mf.READ_ONLY, size=us.nbytes)
        self.VS = cl.Buffer(ctx, mf.READ_ONLY, size=us.nbytes)
        
        self.RO = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes)
        self.GO = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes)
        self.BO = cl.Buffer(ctx, mf.WRITE_ONLY, size=ro.nbytes)

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

    def addCompoundTex(self, xyzn, uvn, vnn, rn, gn, bn):
        p = np.concatenate(xyzn).astype("float32")
        p = align34(p)
        n = np.concatenate(vnn).astype("float32")
        n = align34(n)
        
        uv = np.concatenate(uvn).astype("float32")
        
        tSize = np.array([r.shape[0] for r in rn], dtype="int32")
        rd = np.concatenate([r.reshape((-1,3)) for r in rn]).astype("uint16")
        gd = np.concatenate([r.reshape((-1,3)) for r in gn]).astype("uint16")
        bd = np.concatenate([r.reshape((-1,3)) for r in bn]).astype("uint16")

        self.gSize.append(np.int32(p.shape[0]))
        self.texSize.append(tSize)
        self.useCompound.append(True)

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

    def switch(self, b1, b2, tn):
        vs = np.int32(self.gSize[tn]//BLOCK_SIZE + 1)
        swc.switchBones(cq, (vs, 1), (BLOCK_SIZE, 1),
                   self.BN[tn], np.int8(b1), np.int8(b2),
                   self.gSize[tn], g_times_l=True)
        
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
        self.LI.append(makeRBuf(uv.nbytes//2))

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
        
        return len(self.XYZ)-1

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

    def rotate(self, oldRM, rotMat, origin, cStart, cEnd, tn):
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
        vert.Trotate(cq, (vs, 1), (BLOCK_SIZE, 1),
                    self.XYZ[tn], self.VN[tn],
                    rr, r, o,
                    np.int32(cStart), np.int32(cEnd),
                    g_times_l=True)
        
    def translate(self, coords, cStart, cEnd, tn):
        oo = coords.astype("float32")
        o = makeRBuf(oo.nbytes)
        cl.enqueue_copy(cq, o, oo)
        vs = np.int32((cEnd - cStart)//BLOCK_SIZE + 1)
        vert.Ttranslate(cq, (vs, 1), (BLOCK_SIZE, 1),
                       self.XYZ[tn], o,
                       np.int32(cStart), np.int32(cEnd),
                       g_times_l=True)
        
    def scale(self, origin, scale, tn):
        cStart = 0; cEnd = self.gSize[tn]
        oo = origin.astype("float32")
        o = makeRBuf(oo.nbytes)
        cl.enqueue_copy(cq, o, oo)
        vs = np.int32((cEnd - cStart)//BLOCK_SIZE + 1)
        vert.Tscale(cq, (vs, 1), (BLOCK_SIZE, 1),
                    self.XYZ[tn], o, np.float32(scale),
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
        
    def vertLight(self, lightI, lightD):
        i = lightI.astype("float32")
        d = lightD.astype("float32")
        d = align34(d)
        self.lightInt = makeRBuf(i.nbytes)
        self.lightDir = makeRBuf(d.nbytes * 4//3)
        cl.enqueue_copy(cq, self.lightInt, i)
        cl.enqueue_copy(cq, self.lightDir, d)
        self.numLights = np.int32(lightI.shape[0])
        
        for tn in range(len(self.gSize)):
            vs = np.int32(self.gSize[tn]//BLOCK_SIZE + 1)
            vert.vertL(cq, (vs, 1), (BLOCK_SIZE, 1),
                       self.VN[tn], self.LI[tn],
                       self.lightInt, self.lightDir,
                       self.ambLight,
                       self.numLights, self.gSize[tn],
                       g_times_l=True)

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
                mask=None, shadowIds=[0,1]):
        if mask is None:
            mask = [False] * len(useAlpha)
        gsn = []
        #a = time.time()
        
        for tn in range(len(self.gSize)):
            if mask[tn]:
                gsn.append(0)
                continue
            gs = np.int32(int(self.gSize[tn] / 3 / BLOCK_SIZE)+1)
            gsn.append(gs)
            trisetup.setup(cq, (gs, 1), (BLOCK_SIZE, 1),
                           self.XYZ[tn], self.TI[tn], self.TN[tn],
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
                if useAlpha[tn]:
                    drawA.drawTex(cq, (ns, 1), (BLOCK_SIZE, 1),
                              self.TO[tn],
                             self.RO, self.GO, self.BO,
                             self.DB, self.SP[tn], self.ZZ[tn],
                             self.UV[tn], self.LI[tn], self.XYZ[tn],
                             sm["map"], sm["dim2"], sm["scale"],
                             sm["vec"], sm["pos"],
                             self.ambLight,
                             self.TR[tn], self.TG[tn], self.TB[tn],
                             self.TA[useAlpha[tn] - 1],
                             np.int8(useShadow[tn]),
                             self.W, self.H, np.int32(newSize[tn]),
                             self.texSize[tn],
                             g_times_l=True)
                elif useMip[tn]:
                    drawTriS.drawTex(cq, (ns, 1), (BLOCK_SIZE, 1),
                              self.TO[tn],
                             self.RO, self.GO, self.BO,
                             self.DB, self.SP[tn], self.ZZ[tn],
                             self.UV[tn], self.LI[tn], self.XYZ[tn],
                             sm["map"], sm["dim2"], sm["scale"],
                             sm["vec"], sm["pos"],
                             sm1["map"], sm1["dim2"], sm1["scale"],
                             sm1["vec"], sm1["pos"],
                             self.ambLight,
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
                             self.VN[tn],
                             self.XYZ[tn],
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
                             self.UV[tn], self.LI[tn], self.XYZ[tn],
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
    
    def skydraw(self, sp, u, v):
        gs = int(len(sp) / 3 / BLOCK_SIZE)+1
        
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
    
    def clearShadowMap(self, i):
        sm = self.SHADOWMAP[i]
        s = 4; t = 4
        clearzb.clearFrame(cq, (s, s), (t, t), sm["map"],
                           sm["dim"], sm["dim"],
                           np.int32(t), np.int32(s*t), g_times_l=True)

    def addShadowMap(self, i, size, scale, ambLight):
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
        self.SHADOWMAP[i] = s
        self.ambLight = np.float32(ambLight)

    def placeShadowMap(self, i, pos, facing, ambLight=None):
        sm = self.SHADOWMAP[i]
        p = np.array(pos).astype("float32")
        cl.enqueue_copy(cq, sm["pos"], p)
        f = viewMat(*facing)
        f = align34(f)
        cl.enqueue_copy(cq, sm["vec"], f.astype("float32"))
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
                         self.TO[tn], self.AL, np.int32(i),
                         gs, g_times_l=True)
        
        newSize = np.zeros((self.LT,), dtype="int32")
        cl.enqueue_copy(cq, newSize, self.AL, is_blocking=True)
        nsn = newSize // BLOCK_SIZE + 1

        for i in range(len(tsn)):
            ns = nsn[i]
            tn = tsn[i]
            if newSize[tn] > 0:
                if not useAlpha[tn]:
                    sh.draw(cq, (ns, 1), (BLOCK_SIZE, 1),
                            self.TO[tn],
                            sm["map"], self.SP[tn], self.ZZ[tn],
                            sm["dim"], sm["dim"], np.int32(newSize[i]),
                            g_times_l=True)
                else:
                    shA.draw(cq, (ns, 1), (BLOCK_SIZE, 1),
                             self.TO[tn],
                             sm["map"], self.SP[tn], self.ZZ[tn],
                             self.UV[tn],
                             self.TA[useAlpha[tn] - 1], self.texSize[tn],
                             sm["dim"], sm["dim"], np.int32(newSize[i]),
                             g_times_l=True)
        
    def getSHM(self, i):
        sm = self.SHADOWMAP[i]
        sm = np.empty((sm["dim"], sm["dim"]), dtype="float32")
        cl.enqueue_copy(cq, sm, sm["map"])
        return sm

    def getFrame(self):
        cl.enqueue_copy(cq, self.hro, self.RO, is_blocking=False)
        cl.enqueue_copy(cq, self.hgo, self.GO, is_blocking=False)
        cl.enqueue_copy(cq, self.hbo, self.BO, is_blocking=False)
        cl.enqueue_copy(cq, self.hdb, self.DB)
        
        return (self.hro, self.hgo, self.hbo, self.hdb)
