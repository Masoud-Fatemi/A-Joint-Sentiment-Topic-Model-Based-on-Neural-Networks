# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:38:29 2017
@author: Masoud Fatemi
@organization: IUT, Pattern Analysis and Machine Learning Group
@summary: Implementation of the Replicated Softmax model,
          as presented by R. Salakhutdinov & G.E. Hinton
          in http://www.mit.edu/~rsalakhu/papers/repsoft.pdf
@version: 1.0
baselline.py: baseline sentminet calssification on MR data set

"""

#data_adr = '/home/masoud/Desktop/datasets/mr_v1/pre_test'
data_adr = '/home/masoud/Desktop/datasets/mds/kitchen/test_unigram'
#dict_adr = '/home/masoud/Desktop/datasets/vocab_mr'
lex_adr = '/home/masoud/Desktop/datasets/mpqa'


#load test set, dictionary and sentiment lexicon
TestSet = open(data_adr).read().split('\n')
TestSet = [i for i in TestSet if i.rstrip()]
#dictionary = open(dict_adr).read().split('\n')
#dictionary = [i for i in dictionary if i.rstrip()]

lexicon = open(lex_adr, 'r').read().split('\n')
lexicon = [i for i in lexicon if i.rstrip()]
lexicon = {i.split('\t')[0]:i.split('\t')[1:] for i in lexicon}
del data_adr, lex_adr, i

#seperate test leabels
test_lbls = []
for line in TestSet:
    test_lbls.append(int(line.split(' ')[0]))

#assign detected label by counting sentimenet word
detect_lbls = []
num_pos, num_neg = 0,0 
for line in TestSet:
    tokens = line.split(' ')
    tokens = tokens[1:]
    pos = 0
    neg = 0
    for vocab in tokens:
#        vocab = dictionary[int(token.split(':')[0]) - 1]
        vocab = vocab.split(':')[0]
        if vocab in lexicon:
            if lexicon[vocab][1]>lexicon[vocab][2]:
                pos += 1
            else:
                neg += 1
    if pos>neg:
        num_pos +=1
        detect_lbls.append(1)
    else:
        num_neg +=1
        detect_lbls.append(2)

#caculate error rate on classification               
cnt = 0
for i in range(len(test_lbls)):
    if test_lbls[i] != detect_lbls[i]:
        cnt += 1
        
#show result
print "True Classified: %d from %d" %(len(test_lbls) - cnt,len(test_lbls))
print "Accuracy = %0.2f%%"  %((len(test_lbls) - cnt)/float(len(test_lbls)) * 100)
print "Number of Pos: %d     Number of Neg: %d" %(num_pos, num_neg)
#print "Error rate: %0.2f%%" %((cnt)/float(len(test_lbls)) * 100)
#print "Miss Classified: %d from %d" %(cnt,len(test_lbls))
    