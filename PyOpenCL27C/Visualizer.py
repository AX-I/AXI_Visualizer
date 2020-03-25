# 3D render
# Frontend process (started by backend)
# Multiprocess pipeline
# αβγ

import cv2
fout = "Test.avi"
fc = cv2.VideoWriter_fourcc(*"MJPG")

from tkinter import *
import numpy as np
#import numexpr as ne
import time

import threading
from queue import Empty, Full

import sys, os
import traceback

from PIL import Image, ImageTk, ImageDraw, PngImagePlugin

import pyautogui
def mouseMover(x, y):
    pyautogui.moveTo(x,y)
    
class ThreeDVisualizer(Frame):
    def __init__(self, pipe, eq, infQ,
                 width, height,
                 mouseSensitivity=50,
                 downSample=1, record=False):

        self.record = record # input recording

        # set recVideo = True for video

        if record:
            global VIDOUT
            VIDOUT = cv2.VideoWriter(fout, fc, 30.0, (width, height))

        self.P = pipe
        self.evtQ = eq
        self.infQ = infQ

        root = Tk()
        super().__init__(root)
        
        self.root = root
        self.root.title("3D Visualizer")
        self.downSample = downSample
        self.W = width//downSample
        self.H = height//downSample
        self.W2 = self.W//2
        self.H2 = self.H//2

        self.buffer = np.zeros((self.H, self.W, 3), dtype="float")
        self.nFrames = 0
        self.taa = False

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

        self.frameKeys = {}

    def start(self):
        self.grid(sticky=N+E+S+W)
        self.createCoreWidgets()
        #self.tgFullScreen()

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
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.root.config(background="#000")
        
        self.root.bind("<Escape>", self.escapeMouse)

        self.d = Canvas(self, width=self.W, height=self.H,
                        highlightthickness=0, highlightbackground="black")
        self.d.grid(row=0, column=1, rowspan=10, sticky=N+E+S+W)
        self.d.config(cursor="none", background="#000")

        self.d.bind("<Motion>", self.rotate)
        self.d.bind("<Button-1>", self.setMouse)
        self.d.bind("<B1-Motion>", self.pan)
        
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
        
    def tgFullScreen(self, e=None):
        self.fs = not self.fs
        self.root.attributes("-fullscreen", self.fs)
    
    def customAction(self, a, e):
        if a not in self.frameKeys:
            try: self.evtQ.put_nowait(a)
            except: pass
            self.frameKeys[a] = 1

    def checkPipe(self, cont=True):
        try: action = self.P.get(True, 0.02)
        except Empty: self.empty += 1
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
                    self.d.bind(action[1],
                                lambda x: self.customAction(action[1], x))
                elif action[0] == "screenshot":
                    self.screenshot()
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
        if not self.captureMouse:
            return
        dx = (e.x - self.W2) * self.rotSensitivity
        dy = (e.y - self.H2) * self.rotSensitivity
        self.attractMouse()
        if not self.panning:
            self.sendRot(dx, dy)
        else:
            self.panning = False

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
        i = PngImagePlugin.PngInfo()
        i.add_text("pos", " ".join([str(round(x, 3)) for x in self.pos]))
        i.add_text("dir", " ".join([str(round(x, 3)) for x in self.vv]))
        #self.rawCFrame.save("Screenshots/Screenshot " + ts + ".png", pnginfo=i)
        self.cframe.save("Screenshots/Screenshot " + ts + ".png", pnginfo=i)

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
        self.frameKeys = {}
        self.profileStart()

        rgb = data[0]
        ax = data[1]
        select = data[2]
        self.pos, self.vv = data[3]

        #if (rgb[0,:8] == 0).all(): return
        
        fr = rgb

        self.renderProfile(0)
        
        # Postprocessing goes here
        
        self.renderProfile(1)

##        if self.record:
##            global VIDOUT
##            VIDOUT.write(fr.astype("uint8")[:,:,::-1])
        
##        self.rawCFrame = Image.fromarray(fr.astype("uint8"), "RGB")

        #fr[self.H//2, self.W//2] = 255
##        if self.taa:
##            self.buffer += fr*fr
##            self.nFrames += 1
        
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

        self.renderProfile(3)

        self.cf = ImageTk.PhotoImage(self.cframe)
        self.renderProfile(4)
        
        self.d.tk.call((".!threedvisualizer.!canvas",
                        "itemconfigure", self.finalRender,
                        "-image", self.cf))
        self.renderProfile(5)
    
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
        with open("Message.txt", "w") as f:
            f.write("")
        
        app = ThreeDVisualizer(P, *args)

        app.start()
        app.mainloop()
        time.sleep(1)
        app.finish()
        time.sleep(0.5)
        if app.taa:
            Image.fromarray(np.sqrt(app.buffer / app.nFrames).astype("uint8")).save("Test.png")

    except Exception as e:
        logError(e, "main")
        raise
    finally:
        if app.record:
            VIDOUT.release()
        writeMessage("Frontend profile\n")
        writeMessage("full:" + str(app.full) + ", empty:" + str(app.empty))
        writeMessage(app.printProfile())
        writeMessage("overall spf: " + str(np.sum(app.timfps) / app.numfps))
        #writeMessage("overall spf: " + str(app.totTime / app.frameNum))

def logError(e, message):
    with open("Error.txt", "a") as f:
        f.write("\n" + traceback.format_exc())
        
def writeMessage(m):
    with open("Message.txt", "a") as f:
        f.write(m + "\n")
