#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:26:00 2019

@author: carsault
"""

#%%
import numpy as np
from models.ace_models.utilities import chordUtil
from models.ace_models.utilities.chordVocab import *
from models.ace_models.utilities.chordUtil import *
import torch

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def LoadLabelArr(labfile, dictChord,args, hop_size=512,sr=44100):
    f = open(labfile)
    line = f.readline()
    labarr = np.zeros(round(800*sr/hop_size),dtype="int32")
    while line != "" and line.isspace()==False:
        items = line.split()
        st = int(round(float(items[0])*sr/hop_size))
        ed = int(round(float(items[1])*sr/hop_size))
        lab = dictChord[reduChord(items[2],args.alpha)]
        labarr[st:ed] = lab
        line = f.readline()
    return labarr[:ed]

def transpCQTFrame(cqt,transp = 0):
    transp = 2 * transp
    nbFrame = len(cqt)
    nbBins = len(cqt[0])
    newcqt = torch.zeros(nbFrame, nbBins)
    if transp > 0:
        newcqt[:,0+transp:nbBins] = cqt[:,0:nbBins-transp]
    else:
        newcqt[:,0:nbBins+transp] = cqt[:,0-transp:nbBins]
    return newcqt