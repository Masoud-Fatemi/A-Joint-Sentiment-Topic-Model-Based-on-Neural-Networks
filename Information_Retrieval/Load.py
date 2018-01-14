
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
            else:
                print d
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
    
def dstat_t(file):
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
    
def parse_t (file):
    [docs,words] = dstat_t(file)
    d = 0
    matrix = np.zeros((docs, words))
    with open (file, 'r') as fh:
        for line in fh:
            tokens = line.split()
            tokens = tokens[1:]         
            if len(tokens) > 0:
                for token in tokens:
                    [id,cnt] = token.split(':')
                    v = int(id) - 1
                    c = float(cnt)
                    matrix[d,v] = c
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

def counting_sentiment_lbl(tc, tl, sl, s):
    m = np.zeros((len(tc), s))
    for i in tc:
        for j in range(len(tl)):
            if int(tl[j]) == i:
                if int(sl[j]) == 1:
                    m[i-1][0] += 1
                else:
                    m[i-1][1] += 1
    return m

def counting_topic_lbl_ng(ttl):
    m = { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0,
            11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0,}
    for i in ttl:
        m[int(i)] +=1 
    return m

def counting_topic_lbl_mrmds(ttl):
    m = {1:0, 2:0, 3:0, 4:0, 5:0}
    for i in ttl:
        m[int(i)] +=1
    return m
