# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:38:29 2017
@author: Masoud Fatemi
@organization: IUT, Pattern Analysis and Machine Learning Group
@summary: Implementation of the Replicated Softmax model,
          as presented by R. Salakhutdinov & G.E. Hinton
          in http://www.mit.edu/~rsalakhu/papers/repsoft.pdf
@version: 1.0
load.py: loading sparse feature matrix.

"""

import numpy as np
import pickle

def dstat(file):
    """
    returns documents and words in 'id:cnt' data.
    """
    d,v = 0,0
    with open (file, 'r') as fh:
        for line in fh:
            tokens = line.split()
            tokens = tokens[1:]
            if len(tokens) > 0:
                d = d + 1
                for token in tokens:
                    [id,cnt] = token.split(':')
                    if int(id) > v:
                        v = int(id)
    return (d,v)

def parse (file):
    """
    build a numpy full matrix from sparse 'id:cnt' data.
    """
    [docs,words] = dstat(file)
    d = 0
    matrix = np.zeros((docs, words))
    lbl = []
#    lbl = np.zeros(docs)
    with open (file, 'r') as fh:
        for line in fh:
            tokens = line.split()
            lbl.append(tokens[0])
#            lbl[d] = tokens[0]
            tokens = tokens[1:]
            if len(tokens) > 0:
                for token in tokens:
                    [id,cnt] = token.split(':')
                    v = int(id) - 1
                    c = float(cnt)
                    matrix[d,v] = c
                d = d + 1
    return matrix
#    return matrix, lbl

def pkl(file):
    with open(file) as fh:
        model = pickle.load(fh); fh.close ();
    return model
    
def txt(file):
    data = open(file,'r').read().split('\n')
    data = [i for i in data if i.rstrip()]
    return data
