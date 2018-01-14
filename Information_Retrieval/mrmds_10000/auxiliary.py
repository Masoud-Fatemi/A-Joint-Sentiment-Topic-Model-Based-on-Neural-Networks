import numpy as np
import pickle

def dstat(file):
    d,v = 0,0
    with open (file, 'r') as fh:
        for line in fh:
            tokens = line.split()
            tokens = tokens[2:]
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
    matrix = np.zeros((docs, words + 1))
    with open (file, 'r') as fh:
        for line in fh:
            tokens = line.split()
            lbl = tokens[0]
            tokens = tokens[2:]         
            if len(tokens) > 0:
                for token in tokens:
                    [id,cnt] = token.split(':')
                    v = int(id) - 1
                    c = float(cnt)
                    matrix[d,v] = c
                matrix[d,-1] = int(lbl)
                d = d + 1
    return matrix

def pkl(file):
    with open(file) as fh:
        model = pickle.load(fh); fh.close ();
    return model
    
def txt(file):
    data = open(file,'r').read().split('\n')
    data = [i for i in data if i.rstrip()]
    return data

def lib(file, dict):
    list = []
    for line in file:
        tmp = []
        tokens = line.split()
        tmp.append(tokens[0])
        tokens = tokens[1:]
        for token in tokens:
            if token.split(':')[0] in dict:
                tmp.append(str(dict.index(token.split(':')[0])+1)+':'+token.split(':')[1])
        list.append(' '.join(str(i) for i in tmp))
    return list
    
def sort_lib(file):
    sort = []
    for line in file:
        tmp = []
        tokens = line.split()
        tmp.append(tokens[0])
        tokens = tokens[1:]
        ids = []
        for token in tokens:
            ids.append(int(token.split(':')[0]))
        ids = np.sort(ids)
        for id in ids:
            for token in tokens:
                if str(id) == token.split(':')[0]:
                    tmp.append(str(id)+':'+token.split(':')[1])
                    break
        sort.append(' '.join(str(i)for i in tmp))
    return sort
    
def vec(file):
    np.random.seed(5)
    d = txt(file)
    matrix = np.zeros(len(d))
    for i in range(len(d)):
        if int(d[i]) == 0:
            matrix[i] = np.random.choice([1, 2], p=[1./2, 1./2])
        else:
            matrix[i] = int(d[i])
    return matrix