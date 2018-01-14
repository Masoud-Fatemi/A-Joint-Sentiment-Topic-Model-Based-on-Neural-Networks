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
import Load
import numpy as np
import scipy as sp


def sigmoid(X):
    return (1 + sp.tanh(X/2))/2


model_address = '/home/masoud/Desktop/RS/LearnedModel_mr_wst/model_mr_2000_1000_90'        
test_address = '/home/masoud/Desktop/datasets/mr_v1/mr_2000/test'
sentiment = 2
hidden = int(model_address.split('/')[-1].split('_')[-1])

# load trained model and test set
with open(model_address) as fh:
    model = pickle.load(fh); fh.close ();
w_vh = model['w_vh']
w_sh = model['w_sh']
bias_h = model['bias_h']
bias_s = model['bias_s']
test_set, test_labals = Load.parse(test_address)
del test_address, model, model_address

#calculate label for test set            
detect_label = []
testD  = test_set.sum(axis=1)
h = sigmoid(np.dot(test_set, w_vh) + np.outer(testD, bias_h))
s = np.dot(h, w_sh.T) + bias_s
s = np.exp(s)
sum = s.sum(axis=1)
for i in range(s.shape[0]):
    s[i] = s[i]/float(sum[i])
    
for i in range(s.shape[0]):
    if s[i,0]>s[i,1]:
        detect_label.append(str(1))
    else:
        detect_label.append(str(2))
        
counter = 0
for i in range(len(test_labals)):
    if test_labals[i] != detect_label[i]:
        counter += 1   
      
print "Accuracy = %0.2f%%"  %((test_set.shape[0] - counter)/float(test_set.shape[0]) * 100)

#nn = pkl('/home/masoud/Desktop/RS/sent_calssification/nn_acc')
#pm = pkl('/home/masoud/Desktop/RS/sent_calssification/pm_acc')
#svm_2000 = [67.00] * 9
#svm_10000 = [76.00] * 9 
#import matplotlib.pyplot as plt
#my_xticket = ['10','20','30','40','50','60','70','80','90']
#plt.close()
#plt.figure('10000')
#plt.xlabel('Number of Hidden Units(Topics)')
#plt.ylabel('Accuracy')
#plt.ylim([74, 86])
#plt.xticks(np.array([10,20,30,40,50,60,70,80,90]),my_xticket)
#plt.plot(my_xticket, nn[5],'b-', label='Neural Net')
#plt.plot(my_xticket, nn[7], 'r-o', label='Neural Net with fine-tune')
#plt.plot(my_xticket, svm_10000, 'g^-', label='SVM')
#plt.grid()
#plt.legend()
#plt.show()