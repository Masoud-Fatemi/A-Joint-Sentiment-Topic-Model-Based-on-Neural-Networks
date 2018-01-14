
import scipy as sp
import numpy as np

class RSM(object):
    def train(self, data, units, epochs, iter, lr, weightinit, momentum, btsz):

        dictsize = data.shape[1]
        # initilize weights
        w = weightinit * np.random.randn(dictsize, units)
        bias_v = weightinit * np.random.randn(dictsize)
        bias_h = np.zeros((units))
        # weight updates
        w_update = np.zeros((dictsize, units))
        bias_vu = np.zeros((dictsize))
        bias_hu = np.zeros((units))
        delta = lr/btsz
        batches = data.shape[0]/btsz
        words = np.sum(data)
        preplexity = []
        sum_ppl = []
        for epoch in xrange(epochs):
            lik = 0
            # visit data randomly
            np.random.shuffle(data)
            # gradually increase iter
            for b in xrange(batches):
                start = b * btsz 
                v1 = data[start : start+btsz]
                # hidden biases scaling factor
                D = v1.sum(axis=1)
                # project into hidden
                h1 = sigmoid((np.dot(v1, w) + np.outer(D, bias_h)))
                v2 = v1; h2 = h1
                for i in xrange(iter):
                    (v2,h2,z) = ContrastiveDivergence(v2,h2,w,bias_v,bias_h,D)
                    if i == 0:
                        lik += z
                # compute updates
                w_update = w_update * momentum + np.dot(v1.T, h1) - np.dot(v2.T, h2)
                bias_vu = bias_vu * momentum + v1.sum(axis=0) - v2.sum(axis=0)
                bias_hu = bias_hu * momentum + h1.sum(axis=0) - h2.sum(axis=0)
                # update 
                w += w_update * delta 
                bias_v += bias_vu * delta
                bias_h += bias_hu * delta
            ppl = np.exp (- lik / words)
            print "Epoch[%2d] : PPL = %.02f [iter=%d]" % (epoch, ppl,iter)
            if (epoch+1) == 10:
               preplexity.append(np.mean(np.array(sum_ppl)))
               sum_ppl *= 0
        return { "w"       : w, 
                 "bias_v"  : bias_v, 
                 "bias_h"  : bias_h,
                 "rate"    : lr,
                 "iter"    : iter,
                 "batch"   : btsz,
                 "epoch"   : epochs,
                 "init"    : weightinit,
                 "moment"  : momentum,
                 "ppl"     : ppl,
                 "preplexity" : preplexity
               }

def  ContrastiveDivergence(v1,h1,w,bias_v,bias_h,D):
    likelihood = 0
    btsz = v1.shape[0]
    # project into visible
    v2 = np.dot(h1, w.T) + bias_v
    tmp = np.exp(v2)
    sum = tmp.sum(axis=1)
    sum = sum.reshape((btsz,1))
    v2pdf = tmp / sum
    # perplexity
    likelihood += np.nansum(v1 * np.log(v2pdf))
    # sample from multinomial
    v2 *= 0
    for i in xrange(btsz):
        v2[i] = np.random.multinomial(D[i],v2pdf[i],size=1)
    # project into hidden
    h2 = sigmoid(np.dot(v2, w) + np.outer(D, bias_h))
    return (v2,h2,likelihood)

def sigmoid(X):
    return (1 + sp.tanh(X/2))/2