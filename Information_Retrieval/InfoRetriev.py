
import numpy as np
import copy
import Load
import sys
import matplotlib.pyplot as plt


def lists(idx):
    un = similarity[idx].tolist()
    s = copy.deepcopy(un)
    s.sort(reverse=True)
    return un, s

def precision(tp, t, i):
    (un, s) = lists(i)
    counter = 0
    prec = 0
    for rplus in range(len(s)):
        if int(train_lbl[un.index(s[rplus])]) == t:
            counter += 1
            if counter == tp:
                prec = tp/float(rplus + 1)
                break
    return prec
    
def ppr100(l, k):
    (un, s) = lists(k)
    prec = 0
    for rplus in range(1, len(s)+1):
        if int(train_lbl[un.index(s[-rplus])]) == l:
           rplus = (len(s) - rplus) + 1 
           prec = train_cnt[topic]/float(rplus)
           break
    return prec
          
# Load Data
print 'loading data...',; sys.stdout.flush()
train_lbl_adr = 'mrmds_2000/train_lbl'
test_lbl_adr  = 'mrmds_2000/test_lbl'
sim_adr = 'mrmds_2000/similarity_st_50/sim'
similarity = Load.pkl(sim_adr); del sim_adr
train_lbl = Load.txt(train_lbl_adr); del train_lbl_adr
test_lbl = Load.txt(test_lbl_adr); del test_lbl_adr
print 'done'

#counting number of each class in train and test
print 'counting number of each class...',; sys.stdout.flush()
#train_cnt = { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0,
#            11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0,}
#for i in train_lbl:
#    train_cnt[int(i)] +=1    
#test_cnt = { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0,
#            11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0,}
#for i in test_lbl:
#    test_cnt[int(i)] +=1
train_cnt = { 1:0, 2:0, 3:0, 4:0, 5:0}
for i in train_lbl:
    train_cnt[int(i)] +=1    
test_cnt = { 1:0, 2:0, 3:0, 4:0, 5:0}
for i in test_lbl:
    test_cnt[int(i)] +=1    
print 'done'; del i

# caculation precision and recall
print 'caculation precision and recall'
#recall = 0.016
recalls = np.array([0.0002, 0.00045, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032,
                    0.064, 0.128, 0.256, 0.512, 1])
precisions = []
for recall in recalls:
    total = 0
    for topic in train_cnt:
        ppclass = 0
        tmp = 0
        for index in range(len(test_lbl)):
            if int(test_lbl[index])== topic:
                    ppdata = 0
                    if recall != 1:
                        true_pos = np.floor(recall * train_cnt[topic])+1
                        ppdata = precision(true_pos, topic, index)
                        tmp += ppdata
                    else:
                        ppdata = ppr100(topic, index)
                        tmp += ppdata
    
        ppclass = tmp/float(test_cnt[topic])
        total += ppclass        
        print ('class %d:    Recall = %0.2f%%    Precision = %0.2f%%' %(topic, recall*100, ppclass*100))
    totalpr= total/float(len(train_cnt))   
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
    

    
    