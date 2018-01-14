
import numpy as np
import copy
import Load
from Load import  counting_sentiment_lbl as csl
from Load import  counting_topic_lbl_ng as ctln
from Load import  counting_topic_lbl_mrmds as ctlm
import sys


def lists(idx):
    un = similarity[idx].tolist()
    s = copy.deepcopy(un)
    s.sort(reverse=True)
    return un, s

def precision(tp, top, sen, i):
    (un, s) = lists(i)
    counter = 0
    prec = 0
    for rplus in range(len(s)):
        if int(train_topic_lbl[un.index(s[rplus])]) == top and int(train_sent_lbl[un.index(s[rplus])]) == sent:
            counter += 1
            if counter == tp:
                prec = tp/float(rplus + 1)
                break
    return prec
    
def ppr100(l, m, k):
    (un, s) = lists(k)
    prec = 0
    for rplus in range(1, len(s)+1):
        if int(train_topic_lbl[un.index(s[-rplus])]) == l and int(train_sent_lbl[un.index(s[-rplus])]) == m:
           rplus = (len(s) - rplus) + 1 
           prec = train_topic_cnt[topic]/float(rplus)
           break
    return prec
 

num_of_sentiment = 2        
# Load Data
print 'loading data...'
train_topic_lbl_adr = 'ng/train_topic_lbl'
test_topic_lbl_adr  = 'ng/test_topic_lbl'
train_sent_lbl_adr = 'ng/train_sent_lbl'
test_sent_lbl_adr  = 'ng/test_sent_lbl'
sim_adr = 'ng/similarity_st/sim_50'
similarity = Load.pkl(sim_adr); del sim_adr
train_topic_lbl = Load.txt(train_topic_lbl_adr); del train_topic_lbl_adr
test_topic_lbl = Load.txt(test_topic_lbl_adr); del test_topic_lbl_adr
train_sent_lbl = Load.txt(train_sent_lbl_adr); del train_sent_lbl_adr
test_sent_lbl = Load.txt(test_sent_lbl_adr); del test_sent_lbl_adr

#counting number of each topic and sentiment in train and test
print 'counting number of each topic...'
train_topic_cnt = ctln(train_topic_lbl)   
test_topic_cnt = ctln(test_topic_lbl);
#train_topic_cnt = ctlm(train_topic_lbl)   
#test_topic_cnt = ctlm(test_topic_lbl)

print 'counting number of sentiment for each topic...'
train_sent_cnt = csl(train_topic_cnt, train_topic_lbl, train_sent_lbl, num_of_sentiment)
test_sent_cnt = csl(test_topic_cnt, test_topic_lbl, test_sent_lbl, num_of_sentiment)

# caculation precision and recall
print 'caculation precision and recall'
recalls = np.array([0.0002, 0.00045, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032,
                    0.064, 0.128, 0.256, 0.512, 1])
precisions = []
for recall in recalls:
    total = 0
    for topic in train_topic_cnt:
        temp = 0
        ppclass = 0
        for sent in range(1,3):
            tmp = 0
            for index in range(len(test_topic_lbl)):
                if int(test_topic_lbl[index]) == topic and int(test_sent_lbl[index]) == sent :
                        ppdata = 0
                        if recall != 1:
                            true_pos = np.floor(recall * (train_sent_cnt[topic-1][sent-1]))+1
                            ppdata = precision(true_pos, topic, sent, index)
                            tmp += ppdata
                        else:
                            ppdata = ppr100(topic, sent, index)
                            tmp += ppdata
            temp =+ tmp/float(test_sent_cnt[topic-1][sent-1])
        ppclass = temp/float(2)
        total += ppclass        
        print ('class %d:    Recall = %0.2f%%    Precision = %0.2f%%' %(topic, recall*100, ppclass*100))
    totalpr= total/float(len(train_topic_cnt))   
    precisions.append(totalpr)
    print ('Recall = %0.2f%%    Precision = %0.2%%f' %(recall*100, totalpr*100))
    print "=============="


#import pickle as pkl
#import numpy as np
#import matplotlib.pyplot as plt
#with open('/home/masoud/Desktop/RS/IR/mrmds_10000/similarity_st/p13', 'r') as f:
#    p_st = pkl.load(f)
#with open('/home/masoud/Desktop/RS/IR/mrmds_10000/similarity_t/p13', 'r') as f:
#    p_t = pkl.load(f)
#my_xticks = ['0.02', '0.045', '0.1', '0.2', '0.4', '0.8', '1.6',
#            '3.2', '6.4', '12.8', '25.6', '51.2', '100']
#plt.close()
#plt.figure()
#plt.xlabel('Recall(%)')
#plt.ylabel('Precision(%)')
#plt.xticks(np.arange(0,13),my_xticks)
#plt.plot( p_st, 'r^-', label='proposed model')
#plt.plot( p_t, 'b*-', label='RS')
#plt.grid()
#plt.legend()
#plt.show()

    
    
