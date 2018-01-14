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
import copy
import matplotlib.pyplot as plt
import os
import Load
import e

def extract_topics(m, w):
    temp = [[]for j in range(m.shape[1])]
    lst = [[]for j in range(m.shape[1])]
    for j in range(m.shape[1]):
        temp[j] = m[:,j].tolist()
        lst[j] = m[:,j].tolist()
        lst[j].sort(reverse=True)
    max_word = [[]for j in range(m.shape[1])]
    for j in range(m.shape[1]):
        for i in range(w):
            max_word[j].append(temp[j].index(lst[j][i]))
    return max_word

def show_topics(m, w, d, lst):
    t = []
    for j in range(m.shape[1]):
        print 'Hidden Unit %d:' %(j+1)
        t.append('Hidden Unit '+str(j+1)+':')
        for i in range(w):
            t.append(d[lst[j][i]])
            print "%s"  %(d[lst[j][i]])
        print '\n'
        t.append('*****')
    return t

def show_sentiment_topics(index, topics, d, w):
    st = []
    for i in range(len(index)):        
        for j in range(len(index[i])):
            if i == 0:
                print "positive topic %d:" %(j+1)
                st.append('positive topic '+ str(j+1)+':')
                for k in range(w):
                    print d[topics[index[i][j]][k]]
                    st.append(d[topics[index[i][j]][k]])
                print "======"
                st.append('*****')
            if i == 1:
                print "negative topic %d:" %(j+1)
                st.append('negative topic '+ str(j+1)+':')
                for k in range(w):
                    print d[topics[index[i][j]][k]]
                    st.append(d[topics[index[i][j]][k]])
                print "======"
                st.append('*****')
        print "********"
        print "********"
        st.append('+++++')
        st.append('=====')
    return st

#hidden = 10     

#st_model_adr = 'LearnedModel_mr_st_10000/1000/new/model_mr_10000_1000_'+ str(hidden)
dictionary_adr = '/home/masoud/Desktop/datasets/vocab_ng'
lexicon_adr = '/home/masoud/Desktop/datasets/mpqa'
ntopic = 5
itration = '200'
size = '2000'
hiddens = [i for i in range(ntopic,55,5)] + [60,70,80,90]
#hidden = 2
version = 'new old'
t = []
for hidden in hiddens:
    acc_hidden = []
#    for itr in itration.split(' '):
    for vrs in version.split(' '):
        model_adr = 'LearnedModel_mr_st_'+size+'/'+itration+'/'+vrs+'/'+'model_mr_'+size+'_'+itration+'_'+str(hidden)           
        if os.path.exists(model_adr):    
            # load learned model, dictionary and sentiment lexicon
            sent_model = Load.pkl(model_adr)
            w_vh = sent_model['w_vh']
            w_sh = sent_model['w_sh']
            dictionary = Load.txt(dictionary_adr)
            lexicon = Load.txt(lexicon_adr)
            lexicon = {i.split('\t')[0]:i.split('\t')[1:] for i in lexicon}
    
            # calculate positive and negative weights for each topic
            w_weight = []  
            for j in range(w_vh.shape[1]):
                pos, neg = 0, 0
                for k in range(w_vh.shape[0]):
                    word = dictionary[k]
                    if word in lexicon:
                        if lexicon[word][1]>lexicon[word][2]:
                            pos += w_vh[k,j]
                        else:
                            neg += w_vh[k,j]
                w_weight.append((pos/float(1242)) - (neg/float(1872)))
            sorted = copy.deepcopy(w_weight)
            sorted.sort(reverse = True); del k,j,pos,neg,word
            
            pos_hidden = []
            neg_hidden = []
            for i in range(ntopic):
                pos_hidden.append(w_weight.index(sorted[i]))
                neg_hidden.append(w_weight.index(sorted[-(i+1)]))
                
            cnt = 0
            for i in range(ntopic):
                if w_sh[0,pos_hidden[i]]<w_sh[1,pos_hidden[i]]:
                    cnt += 1
                if w_sh[0,neg_hidden[i]]>w_sh[1,neg_hidden[i]]:
                    cnt+= 1
            
            acc = ((2*ntopic) - cnt)/float(2*ntopic)*100
            acc_hidden.append(acc) 
    t.append(max(acc_hidden))


del acc,acc_hidden,cnt,dictionary;del dictionary_adr,hidden
del hiddens,i;t = e.g(t);del itration,lexicon,lexicon_adr,model_adr
del neg_hidden,pos_hidden,sent_model;del size,sorted,version
del vrs,w_sh,w_vh,w_weight
#cnt = 0
#for j in range(len(hidden_lbl)):
#    if hidden_lbl[j] != sum_weight[j]:
#        cnt += 1
##    print "Accuracy: %0.2f%%" %((len(hidden_lbl)-cnt)/float(len(hidden_lbl))*100)
#acc.append((len(hidden_lbl)-cnt)/float(len(hidden_lbl))*100)
#del j, k, pos, neg, word, cnt

#my_xticket = ['5','10','15','20','25','30','35','40','45','50','60','70','80','90']
#plt.close()
#plt.figure('Visualization Based on Weights for Pos/Neg Words on MR(2000)')
#plt.xlabel('Number of Hidden Units(Topics)')
#plt.ylabel('Accuracy')
#plt.xticks(np.array([5,10,15,20,25,30,35,40,45,50,60,70,80,90]),my_xticket)
#plt.plot(my_xticket, acc,        'b-^', label='Accuracy for 200 itr on MR_2000')
##plt.plot(my_xticket, acc_1000, 'g-o', label='Accuracy for 1000 itr on MR_2000')
#plt.grid()
#plt.legend()
#plt.show()

#Topic Model Visualization
#topic_learned = extract_topics(tm_w,words)
#topics = show_topics(tm_w, words, dictionary, topic_learned)
#file = open('visualize/mr_24916_topics_'+str(words), 'w+')
#for line in topics:
#    file.write(line)
#    file.write('\n')
#file.close()
#my_xticket = [str(i) for i in range(ntopic,55,5)]+['60','70','80','90']
#plt.close()
#plt.figure('5 topic visualization for 1000 itr' )
#plt.xlabel('Number of Hidden Units(Topics)')
#plt.ylabel('Accuracy')
#plt.xticks(np.array([i for i in range(5,55,5)] + [60,70,80,90]),my_xticket)
#plt.plot(my_xticket, total_2000_1000_5,'r+-', label='Proposed Model with 2000')
#plt.plot(my_xticket, total_10000_1000_5,'bo-', label='Proposed Model with 10000')
#plt.plot(my_xticket, total_24916_1000_5,'g^-', label='Proposed Model with 24916')
#plt.grid()
#plt.legend()
#plt.show()


#my_xticket = [str(i) for i in range(ntopic,55,5)]+['60','70','80','90']
#plt.figure('Weight Visualization-DifferentNum of Topics-'+str(ntopic)+'in each one in MR'+size)
#plt.xlabel('Number of Hidden Units(Topics)')
#plt.ylabel('Accuracy')
#plt.xticks(np.array([i for i in range(ntopic,55,5)] + [60,70,80,90]),my_xticket)
#plt.plot(my_xticket, total,'b+-')
#plt.grid()
#plt.legend()
#plt.show()
#Sort weight matric between sentiment and hidden layer
#tmp_w_sh = w_sh.tolist()
#sort_w_sh = copy.deepcopy(tmp_w_sh)
#for i in range(len(sort_w_sh)):
#    sort_w_sh[i].sort(reverse=True)
#
##extract topics with maximum weight to sentiment layer  
#topic_index = [[]for i in range(len(sort_w_sh))]
#for i in range(w_sh.shape[0]):
#    for j in range(ntopic):
#        topic_index[i].append(tmp_w_sh[i].index(sort_w_sh[i][j]))
#del i,j, tmp_w_sh, sort_w_sh 
   
#extract words with maximum weight to hidden layer
#topic_learned = extract_topics(w_vh, 50)

#show learned topics for each sentiment with correspond words 
#sent_topic = show_sentiment_topics(topic_index, topic_learned, dictionary, nwords)


#show number of positive and sentiment word for each topic
#flag = True
#for s in range(len(topic_index)):
#    if flag:
#        print "Postive Topics:"
#        flag = False
#    else:
#       print "Negative Topics:" 
#    for j in range(len(topic_index[s])):
#        pos = 0
#        neg = 0
#        for w in range(nwords):
#            vocab = dictionary[topic_learned[topic_index[s][j]][w]]
#            for l in lexicon:
#                if vocab == l:
##                   if lexicon[l][1]>lexicon[l][2]:
#                    if lexicon[l][1]>lexicon[l][0]:
#                        pos += 1
#                    else:
#                        neg += 1
#                    break
#        print "pos words %0.2f" %(pos/float(447))
#        print "neg words %0.2f" %(neg/float(5))
#        print "*****"
#    print "========="


##Save results
#file = open('visualize/mr_10000_sent_topics_'+str(ntopic)+'_'+str(words), 'w+')
#for line in sent_topic:
#    file.write(line)
#    file.write('\n')
#file.close()    
