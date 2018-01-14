import numpy as np
import pickle

def dstat(file):
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
    [docs,words] = dstat(file)
    d = 0
    matrix = np.zeros((docs, words))
    lbl = np.zeros(docs)
    with open (file, 'r') as fh:
        for line in fh:
            tokens = line.split()
            lbl[d] = tokens[0]
            tokens = tokens[1:]
            if len(tokens) > 0:
                for token in tokens:
                    [id,cnt] = token.split(':')
                    v = int(id) - 1
                    c = float(cnt)
                    matrix[d,v] = c
                d = d + 1
    return matrix, lbl
    
def vectorize(m):
    number = int(np.max(m))
    matrix = np.zeros((m.shape[0], number))
    for l in range(len(m)):
        x = int(m[l])
        matrix[l, x-1] = 1
    return matrix
    
def pkl(file):
    with open(file) as fh:
        model = pickle.load(fh); fh.close ();
    return model