# AXI Visualizer: Animator
# rv2

from tkinter import *
from tkinter.messagebox import *

import Select1
import Pose2
import Ops3
import os, sys
import webbrowser
import multiprocessing

LOGO = b'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'

HELPTEXT = """AXI Visualizer - Animator

General Usage

Left-click/drag - pan
Right-click/drag - rotate
ASDW - move
R - rotate light
<F2> - screenshot

================================
Rigging and Skinning

Keys:

E- toggle selection mode
Z - next bone
X - previous bone
1 - larger brush
2 - smaller brush
F - next joint
Shift+F - previous joint
T - toggle joint movement

================================
Posing

Keys:

F- next joint
Shift+F - previous joint
1,2,3 - Rotate along X,Y,Z axes (roll, heading, pitch)
0 (zero) - reset joint rotation
P - rotate entire model
4 - show/hide axes

================================
Copyright AgentX Industries 2019

For more info and sample models see http://axi.x10.mx/Visualizer/Animator.html
For more help contact us at http://axi.x10.mx/Contact.html
"""

e = ("Times", 18)
f = ("Times", 15)
g = ("Times", 12)

if getattr(sys, "frozen", False):
    p = os.path.dirname(sys.executable) + "/"
else:
    p = os.path.dirname(os.path.realpath(__file__)) + "/"

class Start(Frame):
    def __init__(self):
        root = Tk()
        super().__init__(root)
        
        self.root = root
        self.root.title("AXI Visualizer")
        #self.root.iconbitmap("Logo.ico")
        try:
            self.root.iconbitmap(p + "AXI.ico")
        except FileNotFoundError:
            pass
        self.start()
    def start(self):
        self.grid(sticky=N+E+S+W)
        
        self.title = Label(self, text="Welcome to AXI Visualizer: Animator !",
                           font=e)
        self.title.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.logoImg = PhotoImage(data=LOGO)
        self.logo = Label(self, image=self.logoImg)
        self.logo.grid(row=1, column=0, columnspan=2, padx=10, pady=(0,10))

        self.columnconfigure(0, weight=1, uniform="b")
        self.columnconfigure(1, weight=1, uniform="b")
        
        self.save = Button(self, text="Rig and Skin", fg="blue", bg="#bdf",
                           command=self.gorig, font=f)
        self.save.grid(row=2, column=0, sticky=N+S+E+W, ipadx=4, ipady=2)
        self.open = Button(self, text="Pose", fg="blue", bg="#dbf",
                           command=self.gopose, font=f)
        self.open.grid(row=2, column=1, sticky=N+S+E+W, ipadx=4, ipady=2)
        
        self.abt = Button(self, text="About", fg="#080",
                           command=self.about, font=g)
        self.abt.grid(row=3, column=0, sticky=N+S+W, pady=(15,0))
        self.help = Button(self, text="Help", fg="#800",
                           command=self.gethelp, font=g)
        self.help.grid(row=3, column=1, sticky=N+S+E, pady=(15,0))

    def gorig(self, f=None):
        self.root.destroy()
        Select1.run(f)
    def gopose(self, f=None):
        self.root.destroy()
        Pose2.run(f)
    def about(self):
        try:
            self.abtwin.destroy()
        except (AttributeError, TclError):
            pass
        self.abtwin = Toplevel()
        self.abtwin.title("About")
        try:
            self.abtwin.iconbitmap(p + "AXI.ico")
        except FileNotFoundError: pass
        disptext = """AXI Visualizer: Animator
© AgentX Industries 2019
http://axi.x10.mx
"""
        self.alabel = Text(self.abtwin, wrap=WORD, font=g, width=30, height=10)
        self.alabel.insert(1.0, disptext)
        self.alabel["state"] = DISABLED
        self.alabel.pack()

    def gethelp(self):
        try:
            self.helpwin.destroy()
        except (AttributeError, TclError):
            pass
        self.helpwin = Toplevel()
        self.helpwin.title("Help")
        try:
            self.helpwin.iconbitmap(p + "AXI.ico")
        except FileNotFoundError: pass
        disptext = HELPTEXT
        self.hscroll = Scrollbar(self.helpwin)
        self.hscroll.pack(side=RIGHT, fill=Y)
        self.hlabel = Text(self.helpwin, wrap=WORD, font=g, width=36, height=20)
        self.hlabel.insert(1.0, disptext)
        self.hlabel["state"] = DISABLED
        self.hlabel.pack()
        self.hlabel.config(yscrollcommand=self.hscroll.set)
        self.hscroll.config(command=self.hlabel.yview)
        self.getmorehelp = Button(self.helpwin, text="More help", fg="blue", command=self.morehelp)
        self.getmorehelp.pack()

    def morehelp(self):
        webbrowser.open("http://axi.x10.mx/Visualizer/Help.html")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    res=False
    try:
        a = open("Settings.txt")
        for line in a:
            if line[0] in "WH": res=True
    except FileNotFoundError:
        pass
    if not res:
        rtest = Tk()
        SW = rtest.winfo_screenwidth()
        SH = rtest.winfo_screenheight()
        rtest.destroy()
        if (SW <= 2560) & (SH <= 2560):
            width = SW // 2
            height = SH // 2
        else:
            width = SW // 3
            height = SH // 3
        with open("Settings.txt", "w") as a:
            a.write("Settings for AXI Visualizer\n\n\
CL=0:0\nW={}\nH={}".format(width,height))
    
    f = Start()
    if len(sys.argv) > 1:
        e = sys.argv[1][-4:]
        if e == ".avp":
            f.gopose(sys.argv[1])
        elif e == ".obj":
            f.gorig(sys.argv[1])
    else:
        f.mainloop()
