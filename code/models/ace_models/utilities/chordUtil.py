#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:38:02 2017

@author: tristan
"""

"""----------------------------------------------------------------------
-- Tristan Metadata and conv
----------------------------------------------------------------------"""

#%%
from models.ace_models.utilities.chordVocab import *

def getDictChord(alpha):
    '''
    Fonction def

    Parameters
    ----------
    tf_mapping: keras.backend tensor float32
        mapping of the costs for the loss function

    Returns
    -------
    loss_function: function
    '''
    chordList = []
    dictChord = {}
    for v in gamme.values():
        if v != 'N':
            for u in alpha.values():
                if u != 'N':
                    chordList.append(v+":"+u)
    chordList.append('N')
    listChord = list(set(chordList))
    listChord.sort()
    for i in range(len(listChord)):
        dictChord[listChord[i]] = i
    return dictChord, listChord

def getDictKey():
    '''
    Fonction def

    Parameters
    ----------
    tf_mapping: keras.backend tensor float32
        mapping of the costs for the loss function

    Returns
    -------
    loss_function: function
    '''
    chordList = []
    dictChord = {}
    for v in gammeKey.values():
        chordList.append(v)
    chordList.append('N')
    listChord = list(set(chordList))
    listChord.sort()
    for i in range(len(listChord)):
        dictChord[listChord[i]] = i
    return dictChord, listChord

#dictA0 = getDictChord(a3)

def reduChord(initChord, alpha= 'a1', transp = 0):
    '''
    Fonction def

    Parameters
    ----------
    tf_mapping: keras.backend tensor float32
        mapping of the costs for the loss function

    Returns
    -------
    loss_function: function
    '''    
    if initChord == "":
        print("buuug")
    initChord, bass = initChord.split("/") if "/" in initChord else (initChord, "")
    root, qual = initChord.split(":") if ":" in initChord else (initChord, "")
    root, noChord = root.split("(") if "(" in root else (root, "")
    qual, additionalNotes = qual.split("(") if "(" in qual else (qual, "")  
    
    root = gamme[root]
    if transp > 0:
        for i in range(transp):
            root = tr[root]
    else:
        for i in range(12+transp):
            #print("transpo")
            root = tr[root]
    
    if qual == "":
        if root == "N" or noChord != "":
            finalChord = "N"
        else:
            finalChord = root + ':maj'
    
    elif root == "N":
        finalChord = "N"
    
    else:
        if alpha == 'a1':
                qual = a1[qual]
        elif alpha == 'a0':
                qual = a0[qual]
        elif alpha == 'a2':
                qual = a2[qual]
        elif alpha == 'a3':
                qual = a3[qual]
        elif alpha == 'a5':
                qual = a5[qual]
        elif alpha == 'reduceWOmodif':
                qual = qual
        else:
                print("wrong alphabet value")
                qual = qual
        if qual == "N":
            finalChord = "N"
        else:
            finalChord = root + ':' + qual

    return finalChord
