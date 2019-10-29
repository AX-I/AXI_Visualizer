# 3D render
# Frontend process

# New: see Select, Pose

from tkinter import *
from tkinter.filedialog import *

import numpy as np
import numexpr as ne
import time

import threading
from queue import Empty, Full

import sys, os
if getattr(sys, "frozen", False):
    p = os.path.dirname(sys.executable) + "/"
else:
    p = os.path.dirname(os.path.realpath(__file__)) + "/"

import traceback

from PIL import Image, ImageTk, ImageDraw

import pyautogui
def mouseMover(x, y):
    pyautogui.moveTo(x,y)
    
class ThreeDVisualizer(Frame):
    def __init__(self, pipe, eq, infQ,
                 width, height,
                 mouseSensitivity=50,
                 downSample=1, guiType=None):

        self.guiType = guiType
        self.P = pipe
        self.evtQ = eq
        self.infQ = infQ

        root = Tk()
        super().__init__(root)
        
        self.root = root
        self.root.title("3D Visualizer")
        
        try:
            self.root.iconbitmap(p + "AXI.ico")
        except FileNotFoundError:
            pass
        
        self.downSample = downSample
        self.W = width//downSample
        self.H = height//downSample
        self.W2 = self.W//2
        self.H2 = self.H//2

        self.α = 0
        self.β = 0
        
        self.panning = False
        self.rx, self.ry = 0, 0
        self.threads = []
        self.captureMouse = True
        self.handles = {}
        self.rotSensitivity = mouseSensitivity / 20000
        self.panSensitivity = mouseSensitivity / 500

        self.empty = 0
        self.full = 0
        self.fs = False

        self.dirs = [False, False, False, False]
        
        self.timfps = np.zeros(6)
        self.numfps = 0

        self.xyselect = np.array((0,0))

    def start(self):
        self.grid(sticky=N+E+S+W)
        self.createCoreWidgets()

        self.evtQ.put(["ready"])
        
        self.pipeLoop = self.d.after_idle(self.checkPipe)
        self.timeStart = time.time()
        self.totTime = 0
        self.frameNum = 0

    def profileStart(self):
        self.profileTime = time.time()
        self.numfps += 1
    def renderProfile(self, n):
        self.timfps[n] += time.time() - self.profileTime

    def createCoreWidgets(self):
        self.root.config(background="#000")
        
        self.root.bind("<Escape>", self.escapeMouse)

        self.d = Canvas(self, width=self.W, height=self.H,
                        highlightthickness=0, highlightbackground="black")
        self.d.grid(row=0, column=4, rowspan=20, sticky=N+E+S+W)
        self.d.config(background="#000")

        self.d.bind("<Button-3>", self.setMouse)
        self.d.bind("<B3-Motion>", self.rotate)
        self.d.bind("<Button-1>", self.setMouse)
        self.d.bind("<B1-Motion>", self.pan)
        self.d.bind("<Control-B1-Motion>", self.rotate)
        
        self.d.bind("<KeyPress-d>", self.moveR)
        self.d.bind("<KeyPress-a>", self.moveL)
        self.d.bind("<KeyPress-w>", self.moveU)
        self.d.bind("<KeyPress-s>", self.moveD)
        self.d.bind("<KeyRelease-d>", self.zeroH)
        self.d.bind("<KeyRelease-a>", self.zeroH)
        self.d.bind("<KeyRelease-w>", self.zeroV)
        self.d.bind("<KeyRelease-s>", self.zeroV)
        
        self.d.bind("q", self.tgMouseCap)
        self.d.bind("<F2>", self.screenshot)
        
        self.root.bind("<F11>", self.tgFullScreen)
        
        self.d.focus_set()

        self.finalRender = self.d.create_image((self.W/2, self.H/2))

        self.save = Button(self, text="Save Model", command=self.saveScene)
        self.save.grid(row=1, column=0, columnspan=3)
        self.open = Button(self, text="Open Model", command=self.openScene)
        self.open.grid(row=2, column=0, columnspan=3)

        if self.guiType == "Select": self.createWidgetsS()
        if self.guiType == "Pose": self.createWidgetsP()

        self.riglb = Listbox(self)
        self.riglb.grid(row=16, column=0, columnspan=3)
        self.riglb.bind("<<ListboxSelect>>", self.sendRN)

    def createWidgetsS(self):
        self.cslabel = Label(self, text="Scale")
        self.cslabel.grid(row=3, column=0, sticky=S, columnspan=3)
        self.cscale = Entry(self, width=5)
        self.cscale.insert(0, 1.0)
        self.cscale.grid(row=4, column=0, sticky=N, columnspan=3)
        self.cscale.bind("<Return>", self.scaleChar)

        self.rtlabel = Label(self, text="Rotate")
        self.rtlabel.grid(row=5, column=0, sticky=S, columnspan=3)
        
        self.rxbt = Button(self, text="X", command=self.rotCX)
        self.rxbt.grid(row=6, column=0, sticky=N)
        self.rybt = Button(self, text="Y", command=self.rotCY)
        self.rybt.grid(row=6, column=1, sticky=N)
        self.rzbt = Button(self, text="Z", command=self.rotCZ)
        self.rzbt.grid(row=6, column=2, sticky=N)

        self.blabel = Label(self, text="---")
        self.blabel.grid(row=7, column=0, columnspan=3)
        self.bindb = Button(self, text="Bind Bone to Joint", command=self.bindj)
        self.bindb.grid(row=8, column=0, columnspan=3)
        self.jlabel = Label(self, text="---")
        self.jlabel.grid(row=9, column=0, columnspan=3)
        self.newj = Button(self, text="New Joint", command=self.newBone)
        self.newj.grid(row=10, column=0, columnspan=3)

    def createWidgetsP(self):
        self.saveS = Button(self, text="Export Scene", command=self.saveAll)
        self.saveS.grid(row=3, column=0, columnspan=3)
        
        self.saveP = Button(self, text="Save Pose", command=self.savePose)
        self.saveP.grid(row=4, column=0, columnspan=3)
        self.openP = Button(self, text="Load Pose", command=self.openPose)
        self.openP.grid(row=5, column=0, columnspan=3)

        self.saveR = Button(self, text="Save Project", command=self.saveProj)
        self.saveR.grid(row=6, column=0, columnspan=3)
        self.openR = Button(self, text="Load Project", command=self.openProj)
        self.openR.grid(row=7, column=0, columnspan=3)
        
        self.prop = Button(self, text="Add object", command=self.openObj)
        self.prop.grid(row=8, column=0, columnspan=3, sticky=S)
        
        self.proplb = Listbox(self, height=3)
        self.proplb.grid(row=9, column=0, columnspan=3, sticky=N)
        self.proplb.bind("<<ListboxSelect>>", self.sendPN)
        self.cslabel = Label(self, text="Scale")
        self.cslabel.grid(row=10, column=0, sticky=S, columnspan=3)
        self.cscale = Entry(self, width=5)
        self.cscale.insert(0, 1.0)
        self.cscale.grid(row=11, column=0, sticky=N, columnspan=3)
        self.cscale.bind("<Return>", self.scaleChar)

        self.rlabel = Label(self, text="---")
        self.rlabel.grid(row=12, column=0, columnspan=3)

    def saveAll(self):
        fname = asksaveasfilename(defaultextension=".obj",
                                  filetypes=[("Wavefront OBJ file", ".obj")],
                                  title="Export Scene")
        self.sendKey("asave"+fname)
    def saveScene(self):
        fname = asksaveasfilename(defaultextension=".obj",
                                  filetypes=[("Wavefront OBJ file", ".obj")],
                                  title="Save Character")
        self.sendKey("save"+fname)
    def openScene(self):
        fname = askopenfilename(defaultextension=".obj",
                                filetypes=[("Wavefront OBJ file", ".obj")],
                                title="Open Character")
        self.sendKey("open"+fname)
        
    def saveProj(self):
        fname = asksaveasfilename(defaultextension=".avp",
                                  filetypes=[("AXI Visualizer Project", ".avp")],
                                  title="Save Project")
        self.sendKey("rsave"+fname)
    def openProj(self):
        fname = askopenfilename(defaultextension=".avp",
                                filetypes=[("AXI Visualizer Project", ".avp")],
                                title="Open Project")
        self.sendKey("ropen"+fname)
        
    def scaleChar(self, e):
        self.sendKey("scale"+self.cscale.get())
    def rotCX(self):
        self.sendKey("rotX")
    def rotCY(self):
        self.sendKey("rotY")
    def rotCZ(self):
        self.sendKey("rotZ")
    def newBone(self):
        self.sendKey("newBone")
    def bindj(self):
        self.sendKey("bindj")

    def sendRN(self, e=None):
        n = self.rigDisplay[int(self.riglb.curselection()[0])][1]
        self.sendKey("switch" + str(n))

    def showProps(self, p):
        self.proplb.delete(0, END)
        for i in p:
            self.proplb.insert(END, i)
        
    def sendPN(self, e=None):
        n = self.proplb.get(self.proplb.curselection()[0])
        self.sendKey("pswitch" + n)

    def savePose(self):
        fname = asksaveasfilename(defaultextension=".pose",
                                  filetypes=[("AXI Animator Pose", ".pose")],
                                  title="Save Pose")
        self.sendKey("psave"+fname)
    def openPose(self):
        fname = askopenfilename(defaultextension=".pose",
                                filetypes=[("AXI Animator Pose", ".pose")],
                                title="Load Pose")
        self.sendKey("popen"+fname)
        
    def openObj(self):
        fname = askopenfilename(defaultextension=".obj",
                                filetypes=[("Wavefront OBJ file", ".obj")])
        self.sendKey("oopen"+fname)
        
    def tgFullScreen(self, e=None):
        self.fs = not self.fs
        self.root.attributes("-fullscreen", self.fs)
    
    def customAction(self, e):
        try:
            self.evtQ.put_nowait(self.handles[str(e.char)])
        except:
            pass

    def showNum(self, b, j, t):
        self.blabel.config(text="Bone " + str(b))
        self.jlabel.config(text="Joint " + str(j) + " /" + str(t-1))

    def showRigLb(self, r, n=None):
        self.riglb.delete(0, END)
        for i in r:
            self.riglb.insert(END, i[0])
        if n is not None:
            ix = 0
            for j in range(len(r)):
                if r[j][1] == n:
                    ix = j
            self.riglb.activate(ix)
            self.riglb.select_set(ix)
            self.riglb.see(ix)
        self.rigDisplay = r

    def showRot(self, r):
        self.rlabel.config(text="Rotation (deg):\n" + \
                           str([int(x*180/3.14) for x in r]))

    def checkPipe(self, cont=True):
        try:
            action = self.P.get(True, 0.02)
        except Empty:
            self.empty += 1
            pass
        else:
            try:
                if action is None:
                    cont = False
                    self.quit()
                elif action[0] == "render":
                    self.render(action[1])
                elif action[0] == "title":
                    self.root.title(action[1])
                elif action[0] == "bg":
                    self.d.config(background=action[1])
                elif action[0] == "key":
                    self.d.bind(action[1], self.customAction)
                    self.handles[action[1]] = action[2]
                elif action[0] == "screenshot":
                    self.screenshot()
                elif action[0] == "bnum":
                    self.showNum(*action[1])
                elif action[0] == "rnum":
                    self.showRigLb(action[1], action[2])
                elif action[0] == "rots":
                    self.showRot(action[1])
                elif action[0] == "pnum":
                    self.showProps(action[1])
            except Exception as e:
                logError(e, "checkPipe")
                cont = False
                self.quit()
        if cont:
            self.pipeLoop = self.d.after(4, self.checkPipe)
        self.frameNum += 1
        self.totTime += time.time() - self.timeStart
        self.timeStart = time.time()

    def setMouse(self, e):
        self.rx = e.x
        self.ry = e.y
        self.d.focus_set()
    def escapeMouse(self, e):
        self.captureMouse = False
    def tgMouseCap(self, e):
        self.captureMouse = not self.captureMouse
        
    def attractMouse(self):
        mx = self.d.winfo_rootx() + self.W2
        my = self.d.winfo_rooty() + self.H2

        t = threading.Thread(target=mouseMover, args=(mx, my))
        t.start()
        self.threads.append(t)
        
    def rotate(self, e):
        #if not self.captureMouse:
        #    return
        dx = (e.x - self.rx) * self.rotSensitivity
        dy = (e.y - self.ry) * self.rotSensitivity
        self.rx = e.x
        self.ry = e.y
        #self.attractMouse()
        #if not self.panning:
        self.sendRot(dx, dy)
        #else:
        #    self.panning = False

    def pan(self, e):
        dx = (e.x - self.rx) * self.panSensitivity
        dy = (e.y - self.ry) * self.panSensitivity
        self.rx = e.x
        self.ry = e.y
        self.sendPan((dx, dy))
        self.panning = True

    def moveU(self, e):
        if not self.dirs[0]:
            self.dirs[0] = True
            self.sendKey("u")
    def moveD(self, e):
        if not self.dirs[1]:
            self.dirs[1] = True
            self.sendKey("d")
    def moveR(self, e):
        if not self.dirs[2]:
            self.dirs[2] = True
            self.sendKey("r")
    def moveL(self, e):
        if not self.dirs[3]:
            self.dirs[3] = True
            self.sendKey("l")
    def zeroV(self, e):
        if self.dirs[0] or self.dirs[1]:
            self.dirs[0] = False
            self.dirs[1] = False
            self.sendKey("ZV")
    def zeroH(self, e):
        if self.dirs[2] or self.dirs[3]:
            self.dirs[2] = False
            self.dirs[3] = False
            self.sendKey("ZH")

    def screenshot(self, e=None):
        ts = time.strftime("%Y %b %d %H-%M-%S", time.gmtime())
        self.rawCFrame.save("Screenshots/Screenshot " + ts + ".png")
        #self.cframe.save("Screenshots/Screenshot " + ts + ".png")

    def sendKey(self, key):
        try: self.evtQ.put_nowait(("eventk", key))
        except Full: self.full += 1
    def sendRot(self, r1, r2):
        try: self.evtQ.put_nowait(("event", r1, r2))
        except Full: self.full += 1
    def sendPan(self, r):
        try: self.evtQ.put_nowait(("eventp", r))
        except Full: self.full += 1

    def render(self, data):
        self.profileStart()

        rgb = data[0]
        ax = data[1]
        select = data[2]
        drawcross = data[3]
        lines = data[4]
        
        fr = rgb

        self.renderProfile(0)
        
        # Postprocessing goes here
        
        #oF = self.oldFrame
        #self.oldFrame >>= 4
        #self.oldFrame *= 15
        #self.oldFrame += (fr>>4)
        
        self.renderProfile(1)

##        global VIDOUT
##        VIDOUT.write(fr.astype("uint8")[:,:,::-1])
        self.rawCFrame = Image.fromarray(fr.astype("uint8"), "RGB")
        
        self.renderProfile(2)
        
        if self.downSample > 1:
            c = fr
            for i in range(int(np.log2(self.downSample))):
                b = c[:-1:2] + c[1::2]
                b >>= 1
                c = b[:,:-1:2] + b[:,1::2]
                c >>= 1
            self.cframe = Image.fromarray(c.astype("uint8"), "RGB")
        else:
            self.cframe = Image.fromarray(fr.astype("uint8"), "RGB")

        if lines is not None:
            d = ImageDraw.Draw(self.cframe)
            for i in range(len(lines)//2):
                d.line((*lines[2*i], *lines[2*i+1]), fill=(100,240,250), width=2)
            
        if ax is not None:
            d = ImageDraw.Draw(self.cframe)
            # xyz
            d.line((*ax[0], *ax[1]), fill=(0,0,255), width=3)
            d.line((*ax[0], *ax[2]), fill=(255,0,0), width=3)
            d.line((*ax[0], *ax[3]), fill=(0,255,0), width=3)

        if drawcross:
            d = ImageDraw.Draw(self.cframe)
            d.line((self.W2, self.H2-4, self.W2, self.H2+4),
                   fill=(255,255,255), width=1)
            d.line((self.W2-4, self.H2, self.W2+4, self.H2),
                   fill=(255,255,255), width=1)

        if select:
            self.d.bind("<Motion>", self.drawSelect)
            self.d.bind("<Button-1>", self.sendSelect)
            self.d.bind("<B1-Motion>", self.sendSelect)
            
            d = ImageDraw.Draw(self.cframe)
            d.rectangle((*self.xyselect-2, *self.xyselect+2), fill=(255,0,255))
        else:
            self.d.unbind("<Motion>")
            self.d.bind("<Button-1>", self.setMouse)
            self.d.bind("<B1-Motion>", self.pan)
        self.renderProfile(3)
        
        self.renderProfile(3)

        self.cf = ImageTk.PhotoImage(self.cframe)
        self.renderProfile(4)
        
        self.d.tk.call((".!threedvisualizer.!canvas",
                        "itemconfigure", self.finalRender,
                        "-image", self.cf))
        self.renderProfile(5)

    def drawSelect(self, e):
        self.xyselect = np.array((e.x, e.y))
        try: self.evtQ.put_nowait(("select", (0, self.xyselect)))
        except Full: pass

    def sendSelect(self, e):
        self.xyselect = np.array((e.x, e.y))
        try: self.evtQ.put_nowait(("select", (1, self.xyselect)))
        except Full: pass
    
    def finish(self):
        for t in self.threads:
            t.join()
        try:
            self.root.destroy()
        except TclError:
            pass
        try:
            self.evtQ.put(None, True, 1)
        except (Full, BrokenPipeError):
            pass

    def printProfile(self):
        st = "Av. fps: " + str(np.reciprocal(self.timfps) * self.numfps)
        a = self.timfps / self.numfps
        st += "\nAv. spf: " + str(a)
        st += "\n" + str(a[0])
        for i in range(len(a) - 1):
            st += "\n" + str(a[i + 1] - a[i])
        return st

def runGUI(P, *args): 
    try:
        with open("Log_f.txt", "w") as f:
            f.write("")
        
        app = ThreeDVisualizer(P, *args)

        app.start()
        app.mainloop()
        time.sleep(1)
        app.finish()
        time.sleep(0.5)

    except Exception as e:
        logError(e, "main")
        raise
    finally:
##        VIDOUT.release()
        writeMessage("Frontend profile\n")
        writeMessage("full:" + str(app.full) + ", empty:" + str(app.empty))
        writeMessage(app.printProfile())
        writeMessage("overall spf: " + str(np.sum(app.timfps) / app.numfps))
        #writeMessage("overall spf: " + str(app.totTime / app.frameNum))

def logError(e, message):
    with open("Error.txt", "a") as f:
        f.write("\n" + traceback.format_exc())
        
def writeMessage(m):
    with open("Log_f.txt", "a") as f:
        f.write(m + "\n")
