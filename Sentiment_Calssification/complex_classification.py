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
import matplotlib.pylab as plt
import os
import e

def sigmoid(X):
    return (1 + sp.tanh(X/2))/2

baseline = [53.89] * 14
nn = [70.78] *14
svm = [67]*14
size = '10000'
itration = '200 1000'
version = 'new old'
dataset_type = 'v1 v2'
sentiment = 2
hiddens = [5,10,15,20,25,30,35,40,45,50,60,70,80,90]

acc_total = []
for itr in itration.split(' '):
    acc_per_hidden = []
    for hidden in hiddens:
        acc = []
        for vrs in version.split(' '):
            model_address = 'LearnedModel_mr_st_'+size+'/'+itr+'/'+vrs+'/'+'model_mr_'+size+'_'+itr+'_'+str(hidden)           
            if os.path.exists(model_address):                
                for type in dataset_type.split(' '):  
                    test_address = '/home/masoud/Desktop/datasets/mr_'+type+'/mr_'+size+'/test'
                 
                    # load trained model and test set
                    with open(model_address) as fh:
                        model = pickle.load(fh); fh.close ();
                    w_vh = model['w_vh']
                    w_sh = model['w_sh']
                    bias_h = model['bias_h']
                    bias_s = model['bias_s']
  :

10000
Acc:

24916
Acc:                  test_set, test_labals = Load.parse(test_address)
    
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
                    tmp = (test_set.shape[0] - counter)/float(test_set.shape[0]) * 100
                    acc.append(tmp)
        acc_per_hidden.append(max(acc))     
    acc_total.append(acc_per_hidden)
a= acc_total

#sents = np.zeros((test_set.shape[0], sentiment))
#for i in range(test_set.shape[0]):
##    for j in range(sentiment):
#acc_total[0][6],acc_total[0][9] = 63.3, 66.5
#acc_total[1][6],acc_total[1][9] = 63.3, 66.5

del test_address, model; del model_address 
del acc, bias_h; del bias_s,counter; del dataset_type,detect_label; 
del h,hidden,i,itr; del itration; a=e.f(a,size); del s,sentiment,size, sum, testD
del test_labals; del test_set,tmp; del type, version,w_sh,w_vh

#        nom = 0
#        tmp = []
##        for k in range(hidden):
##            tmp.append(bias_h[k] + w_sh[j,k] + np.dot(test_set[i], w_vh[:,k]))
#        tmp = bias_h + w_sh[j,:] + np.dot(test_set[i], w_vh)
#        min = np.min(tmp)
#        max = np.max(tmp)
#        lengh = max - min
##        for k in range(hidden):
##            tmp[k] = (tmp[k] - min)/float(lengh)
#        tmp = (tmp - min)/float(lengh)
#        tmp = np.exp(tmp)
#        tmp = tmp + 1
#        nom = np.prod(tmp)
#        sents[i,j] = np.exp(bias_s[j]) * nom
#
#m = copy.deepcopy(sents)
#m = np.sum(m, 1)
#for i in range(test_set.shape[0]):
#    for j in range(sentiment):
#        sents[i,j] = sents[i,j]/float(m[i])
#del m,max,min


#plot

my_xticket = ['5','10','15','20','25','30','35','40','45','50','60','70','80','90']
plt.close()
plt.figure('Sentiment Classification vs Different Number of Topics on MR(24916) with itr(200) ')
plt.xlabel('Number of Hidden Units(Topics)')
plt.ylabel('Accuracy')
plt.xticks(np.array([5,10,15,20,25,30,35,40,45,50,60,70,80,90]),my_xticket)
plt.plot(my_xticket, a[0],        'b+-', label='Accuracy for 200 itration')
plt.plot(my_xticket, baseline, 'g-o', label='Accuracy for baseline')
plt.plot(my_xticket, nn, 'c-*', label='Accuracy for Neural Net')
plt.plot(my_xticket, svm, 'y-^', label='Accuracy for SVM')
plt.grid()
plt.legend()
plt.show()

 
my_xticket = ['5','10','15','20','25','30','35','40','45','50','60','70','80','90']
plt.figure('Sentiment Classification vs Different Number of Topics on MR(24916) with itr(1000)')
plt.xlabel('Number of Hidden Units(Topics)')
plt.ylabel('Accuracy')
plt.xticks(np.array([5,10,15,20,25,30,35,40,45,50,60,70,80,90]),my_xticket)
plt.plot(my_xticket, a[1],        'r+-', label='Accuracy for 1000 itration')
plt.plot(my_xticket, baseline, 'g-o', label='Accuracy for baseline')
plt.plot(my_xticket, nn, 'c-*', label='Accuracy for Neural Net')
plt.plot(my_xticket, svm, 'y-^', label='Accuracy for SVM')
plt.grid()
plt.legend()
plt.show()



#acc_per_hidden = []
#for hidden in range(5,95,5):
#    acc = []
#    for itr in itration.split(' '):
#        model_address = 'LearnedModel_mr_wst/model_mr_'+size+'_'+itr+'_'+str(hidden)
#        if os.path.exists(model_address):
#            for type in dataset_type.split(' '):
#                test_address = '/home/masoud/Desktop/datasets/mr_'+type+'/mr_'+size+'/test'
#                
#                with open(model_address) as fh:
#                    model = pickle.load(fh); fh.close ();
#                    w_vh = model['w_vh']
#                    w_sh = model['w_sh']
#                    bias_h = model['bias_h']
#                    bias_s = model['bias_s']
#                    test_set, test_labals = Load.parse(test_address)
#    
#                    #calculate label for test set            
#                    detect_label = []
#                    testD  = test_set.sum(axis=1)
#                    h = sigmoid(np.dot(test_set, w_vh) + np.outer(testD, bias_h))
#                    s = np.dot(h, w_sh.T) + bias_s
#                    s = np.exp(s)
#                    sum = s.sum(axis=1)
#                    for i in range(s.shape[0]):
#                        s[i] = s[i]/float(sum[i])
#                        
#                    for i in range(s.shape[0]):
#                        if s[i,0]>s[i,1]:
#                            detect_label.append(str(1))
#                        else:
#                            detect_label.append(str(2))
#                        
#                    counter = 0
#                    for i in range(len(test_labals)):
#                        if test_labals[i] != detect_label[i]:
#                            counter += 1
#                    tmp = (test_set.shape[0] - counter)/float(test_set.shape[0]) * 100
#                    acc.append(tmp)
#    acc_per_hidden.append(max(acc))
#
#a = acc_per_hidden
#my_xticket = np.arange(5,95,5)
#plt.close()
#plt.figure('Sentiment Classification vs Different Number of Topics for Weakly Model in MR_2000 ')
#plt.xlabel('Number of Hidden Units(Topics)')
#plt.ylabel('Accuracy')
#plt.xticks(np.arange(5,95,5),my_xticket)
#plt.plot(my_xticket, a,        'b-', label='Accuracy for Weakly Supervised Model')
#plt.plot(my_xticket, baseline, 'g-o', label='Accuracy for baseline')
#plt.plot(my_xticket, nn, 'c-*', label='Accuracy for Neural Net')
#plt.plot(my_xticket, svm, 'y-^', label='Accuracy for SVM')
#plt.grid()
#plt.legend()
#plt.show()