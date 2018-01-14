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
    matrix = np.zeros((docs, words+1))
#    matrix = []
    with open (file, 'r') as fh:
        for line in fh:
            tokens = line.split()
            lbl = int(tokens[0])
            tokens = tokens[1:]
            if len(tokens) > 0:
                for token in tokens:
                    [id,cnt] = token.split(':')
                    v = int(id) - 1
                    c = float(cnt)
                    matrix[d,v] = c
                matrix[d, -1] = lbl
                d = d + 1
    return matrix            