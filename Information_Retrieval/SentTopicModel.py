
import scipy as sp
import numpy as np
import pickle

class RSM(object):
    def train(self, data, units, sentiment, epochs, iter, lr, weightinit, momentum, btsz):

        dictsize = data.shape[1] - 1

        # initilize weights
        w_vh = weightinit * np.random.randn(dictsize, units)
        w_sh = weightinit * np.random.randn(sentiment, units)
        bias_v = weightinit * np.random.randn(dictsize)
        bias_s = weightinit * np.random.randn(sentiment)
        bias_h = np.zeros((units))
       
       # weight updates
        w_vh_update = np.zeros((dictsize, units))
        w_sh_update = np.zeros((sentiment, units))
        bias_vu = np.zeros((dictsize))
        bias_su = np.zeros((sentiment))
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
            labels = data[:,-1]
            labels = vectorize(labels)
            datas = np.delete(data, -1, 1)
            # gradually increase iter
            for b in xrange(batches):
                start = b * btsz
                v1 = datas[start : start+btsz]
                s1 = labels[start : start+btsz]
                # hidden biases scaling factor
                D = v1.sum(axis=1)
                # project into hidden
                h1 = sigmoid((np.dot(v1, w_vh) + np.dot(s1, w_sh) + np.outer(D, bias_h)))
                v2 = v1; s2 = s1; h2 = h1
                for i in xrange(iter):
                    (v2,s2,h2,z) = ContrastiveDivergence(v2,s2,h2,w_vh,w_sh,bias_v,bias_s,bias_h,D)
                    if i == 0:
                        lik += z
                # compute updates
                w_vh_update = w_vh_update * momentum + np.dot(v1.T, h1) - np.dot(v2.T, h2)
                w_sh_update = w_sh_update * momentum + np.dot(s1.T, h1) - np.dot(s2.T, h2)
                bias_vu = bias_vu * momentum + v1.sum(axis=0) - v2.sum(axis=0)
                bias_su = bias_su * momentum + s1.sum(axis=0) - s2.sum(axis=0)
                bias_hu = bias_hu * momentum + h1.sum(axis=0) - h2.sum(axis=0)
                # update 
                w_vh += w_vh_update * delta
                w_sh += w_sh_update * delta 
                bias_v += bias_vu * delta
                bias_s += bias_su * delta
                bias_h += bias_hu * delta
            ppl = np.exp (- lik / words)
            print "Epoch[%2d] : PPL = %.02f [iter=%d]" % (epoch, ppl,iter)
            sum_ppl.append(ppl)
            if (epoch+1)%10 ==0:                
                preplexity.append(np.mean(np.array(sum_ppl)))
                sum_ppl *= 0
            if (epoch+1) == 0:
                mini_result = save(w_vh, w_sh, bias_v, bias_s, bias_h, lr, iter, btsz, epochs, weightinit,momentum, ppl, preplexity)
                with open('LearnedModel_ng/model_ng_'+str(epoch), 'wb') as file:
                    pickle.dump (mini_result, file, 1)
        return { "w_vh"       : w_vh, 
                 "w_sh"       : w_sh,
                 "bias_v"     : bias_v,
                 "bias_s"     : bias_s,
                 "bias_h"     : bias_h,
                 "rate"       : lr,
                 "iter"       : iter,
                 "batch"      : btsz,
                 "epoch"      : epochs,
                 "init"       : weightinit,
                 "moment"     : momentum,
                 "ppl"        : ppl,
                 "preplexity" : preplexity
               }

def  ContrastiveDivergence(v1,s1,h1,w_vh,w_sh,bias_v,bias_s,bias_h,D):

    likelihood = 0
    btsz = v1.shape[0]
    
    # project into visible
    v2 = np.dot(h1, w_vh.T) + bias_v
    s2 = np.dot(h1, w_sh.T) + bias_s
    
    tmp = np.exp(v2)
    sum = tmp.sum(axis=1)
    sum = sum.reshape((btsz,1))
    v2pdf = tmp / sum
    tmp = np.exp(s2)
    sum = tmp.sum(axis=1)
    sum = sum.reshape((btsz,1))
    s2pdf = tmp / sum
    
    # perplexity
    likelihood += np.nansum(v1 * np.log(v2pdf))

    #sample from multinomial
    v2 *= 0; s2 *= 0
    for i in xrange(btsz):
        v2[i] = np.random.multinomial(D[i],v2pdf[i],size=1)
        if s2pdf[i,0]>s2pdf[i,1]:
            s2[i,0] =1
        else:
            s2[i,1]=1
    # project into hidden
    h2 = sigmoid((np.dot(v2, w_vh) + np.dot(s2, w_sh) + np.outer(D, bias_h)))
    return (v2,s2,h2,likelihood)

def sigmoid(X):
    return (1 + sp.tanh(X/2))/2
    
def vectorize(m):
    number = int(np.max(m))
    matrix = np.zeros((m.shape[0], number))
    for l in range(len(m)):
        x = int(m[l])
        matrix[l, x-1] = 1
    return matrix

def save(w_vh, w_sh, bias_v, bias_s, bias_h, lr, iter, btsz, epochs, weightinit,momentum, ppl, preplexity):
     return{ 
             "w_vh"       : w_vh, 
             "w_sh"       : w_sh,
             "bias_v"     : bias_v,
             "bias_s"     : bias_s,
             "bias_h"     : bias_h,
             "rate"       : lr,
             "iter"       : iter,
             "batch"      : btsz,
             "epoch"      : epochs,
             "init"       : weightinit,
             "moment"     : momentum,
             "ppl"        : ppl,
             "preplexity" : preplexity
           }