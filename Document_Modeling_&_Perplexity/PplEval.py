# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:38:29 2017
@author: Masoud Fatemi
@organization: IUT, Pattern Analysis and Machine Learning Group
@summary: Implementation of the Replicated Softmax model,
          as presented by R. Salakhutdinov & G.E. Hinton
          in http://www.mit.edu/~rsalakhu/papers/repsoft.pdf
@version: 1.0
"""

import pickle
import numpy as np
import scipy as sp
import Load
#import matplotlib.pylab as plt




def sigmoid(X):
    return (1 + sp.tanh(X/2))/2


model_address_t = 'ppl/LearnedModel_mr_t/model_mr_10000_200'
model_address_st = 'ppl/LearnedModel_mr_st/model_mr_10000_200'
test_address = '/home/masoud/Desktop/datasets/mr_v1/mr_10000/test'

#==============================================================================
# load learned model
#==============================================================================
with open(model_address_t) as fh:
    model = pickle.load(fh); fh.close ();
w_t = model['w']
bias_v_t  = model['bias_v']
bias_h_t = model['bias_h']
with open(model_address_st) as fh:
    model = pickle.load(fh); fh.close ();
w_st = model['w_vh']
bias_v_st  = model['bias_v']
bias_h_st  = model['bias_h']
del model_address_t,model_address_st

#==============================================================================
# load test set
#==============================================================================
#train = Load.parse(train_address)
test = Load.parse(test_address)
del test_address


#trainD = train.sum(axis=1)
testD  = test.sum(axis=1)

n = test.shape[0]


# compute hidden activations
h_t  = sigmoid(np.dot(test, w_t) + np.outer(testD, bias_h_t))
h_st = sigmoid(np.dot(test, w_st) + np.outer(testD, bias_h_st))


# compute visible activations
v_t = np.dot(h_t, w_t.T) + bias_v_t
tmp = np.exp(v_t)
sum = tmp.sum(axis=1)
sum = sum.reshape((n,1))
pdf = tmp / sum
z = np.nansum(test * np.log(pdf))
s = np.sum(test)
ppl_t = np.exp(- z / s)
print "PPL_topic      =", ppl_t

v_st = np.dot(h_st, w_st.T) + bias_v_st
tmp = np.exp(v_st)
sum = tmp.sum(axis=1)
sum = sum.reshape((n,1))
pdf = tmp / sum
z = np.nansum(test * np.log(pdf))
s = np.sum(test)
ppl_st = np.exp(- z / s)
print "PPL_sent/topic =", ppl_st
del bias_h_st,bias_h_t,bias_v_st,bias_v_t
del v_st,v_t
del h_st,h_t
del model, pdf, n, sum, tmp,s,z,test,testD

import matplotlib.pyplot as plt
import pickle
import numpy as np
model_t = 'LearnedModel_mr_t/model_mr_10000(mean_ppl_each_10itr)'
model_st = 'LearnedModel_mr_st/model_mr_10000(mean_ppl_each_10itr)'
with open(model_t) as fh:
    model_topic = pickle.load(fh); fh.close ();
with open(model_st) as fh:
    model_sent_topic = pickle.load(fh); fh.close ();
ppl_t = model_topic['preplexity']
ppl_st = model_sent_topic['preplexity']
my_xticket = np.arange(10, 1010, 10)
plt.close()
plt.figure("model_mr_10000(mean_ppl_each_10itr)")
plt.xlabel('Itration')
plt.ylabel('Preplexity')
plt.plot(my_xticket, ppl_t, 'b-', label='Ppl without Sentiment')
plt.plot(my_xticket, ppl_st, 'r-', label='Ppl with Sentiment')
plt.grid()
plt.legend()
plt.show()