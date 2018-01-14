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


import sys
import Load
import SentTopicModel
import pickle
#import time


#==============================================================================
# default parameters
#==============================================================================
hiddens = 50
sentiment = 2
epochs = 10
iter = 1
rate = 0.001
batch = 4
proto = 1	# binary pickled mode
weightinit = 0.001
momentum = 0.9
data_adr = '/home/masoud/Desktop/datasets/ng/train'
lbl_adr = '/home/masoud/Desktop/datasets/train_sent'

#==============================================================================
# load train data
#==============================================================================
print 'loading data...',; sys.stdout.flush()
data = Load.parse(data_adr)
#data = sLoad.parse(address)
print 'done.'
print 'number of documents        = %d' % data.shape[0]
print 'number of lexicon          = %d' %(data.shape[1]-1)
print 'number of hidden variables = %d' % hiddens
print 'number of sentiment        = %d' % sentiment
print 'number of learning epochs  = %d' % epochs
print 'number of CD iterations    = %d' % iter
print 'minibatch size             = %d' % batch
print 'learning rate              = %g' % rate


#==============================================================================
# Training and save the result of training
#==============================================================================
Sent_Replicated_Softmax = SentTopicModel.RSM()
result = Sent_Replicated_Softmax.train(data, hiddens,
                                       sentiment, epochs, iter, rate, weightinit, momentum, batch)
#with open('LearnedModel_mr_st/model_mr_'+address.split('/')[-2].split('_')[1]+'_'+
# str(epochs)+'_'+str(hiddens), 'wb') as file:
#with open('model', 'wb') as file:
#    pickle.dump (result, file, proto)