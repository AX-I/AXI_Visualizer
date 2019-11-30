# Cubemap for Visualizer

from math import sqrt, sin, cos, pi
import numpy
from Utils import eucDist, anglesToCoords
from PIL import Image, ImageTk
from tkinter import PhotoImage
import json
import numexpr as ne

class CubeMap:
    def __init__(self, tex, setup=1, delraw=True):
        if type(tex) is str:
            ti = Image.open(tex).convert("RGB")
            m = ti.size[1]
            if m*6 != ti.size[0]:
                raise ValueError("Image is not w:h = 6:1!")
            self.m = m
            self.rawtexture = numpy.array(ti).astype("float")
            ta = self.rawtexture
            lum = 0.2626 * ta[:,:,0] + 0.6152 * ta[:,:,1] + 0.1222 * ta[:,:,2]
            ta[lum > 253] *= 4
            ta = ta*ta*(numpy.expand_dims(lum, 2)**(1/4)) / (4*8)
            numpy.clip(ta, None, 256*256-1, ta)
            self.rawtexture = ta
            #self.rawtexture = self.rawtexture * self.rawtexture / 4
            
        elif type(tex) is numpy.ndarray:
            self.rawtexture = numpy.concatenate(tex, axis=1)
            self.m = self.rawtexture.shape[0]
        else:
            raise TypeError("tex is not filename or numpy array!")
            
        self.texture = []
        if setup == 1:
            self.setupTexture()
        elif setup == 2:
            self.setupTex2()
        if delraw:
            del self.rawtexture
        
    def setupTexture(self):
        m = self.m
        for i in range(6):
            self.texture.append(self.rawtexture[:,m*i:m*(i+1)].transpose((1, 0, 2)))
        temp = self.texture[1]
        self.texture[1] = self.texture[4]
        self.texture[4] = self.texture[5]
        self.texture[5] = self.texture[2]
        self.texture[2] = self.texture[0]
        self.texture[0] = temp
        
        self.texture[5] = self.texture[5].transpose((1, 0, 2))
        self.texture[2] = self.texture[2].transpose((1, 0, 2))
        self.texture[1] = self.texture[1].transpose((1, 0, 2))
        self.texture[4] = self.texture[4].transpose((1, 0, 2))
        
        self.texture[0] = numpy.flip(self.texture[0], axis=(0, 1))
        self.texture[1] = numpy.flip(self.texture[1], axis=1)
        self.texture[2] = numpy.flip(self.texture[2], axis=0)
        self.texture[3] = numpy.flip(self.texture[3], axis=0)
        
        self.texture = numpy.array(self.texture)
    
    def setupTex2(self):
        m = self.m
        for i in range(6):
            self.texture.append(self.rawtexture[:,m*i:m*(i+1)])
        temp = numpy.array(self.texture[1])
        self.texture[1] = self.texture[4]
        self.texture[4] = self.texture[5]
        self.texture[5] = self.texture[2]
        self.texture[2] = self.texture[0]
        self.texture[0] = temp
        
        self.texture[5] = self.texture[5].transpose((1, 0, 2))
        self.texture[2] = self.texture[2].transpose((1, 0, 2))
        self.texture[1] = self.texture[1].transpose((1, 0, 2))
        self.texture[4] = self.texture[4].transpose((1, 0, 2))
        
        self.texture[0] = numpy.flip(self.texture[0], axis=(0, 1))
        self.texture[1] = numpy.flip(self.texture[1], axis=0)
        self.texture[2] = numpy.flip(self.texture[2], axis=1)
        self.texture[3] = numpy.flip(self.texture[3], axis=1)
        
        self.texture = numpy.concatenate(self.texture, axis=1)
