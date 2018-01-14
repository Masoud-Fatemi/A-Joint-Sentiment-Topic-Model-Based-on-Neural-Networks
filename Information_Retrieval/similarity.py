
import pickle
import numpy as np
import scipy as sp
import Load
from sklearn.metrics.pairwise import cosine_similarity
#from scipy import spatial


def sigmoid(X):
    return (1 + sp.tanh(X/2))/2
    
def vectorize(m):
    number = int(np.max(m))
    matrix = np.zeros((m.shape[0], number))
    for l in range(len(m)):
        x = int(m[l])
        matrix[l, x-1] = 1
    return matrix
    

train_adr = 'ng/dataset_st/train'
test_adr = 'ng/dataset_st/test'
model_adr = 'ng/LearnedModel_st/model_100'

# load Data and learned model
train = Load.parse(train_adr)
test = Load.parse(test_adr)

train_lbl = train[:,-1]
train_lbl = vectorize(train_lbl)
train = np.delete(train, -1, 1)
test_lbl = test[:,-1]
test_lbl = vectorize(test_lbl)
test = np.delete(test, -1, 1)

with open(model_adr) as fh:
    model = pickle.load(fh); fh.close ();
#w_vh = model['w_vh']
#w_sh = model['w_sh']
w = model['w']
bias_v  = model['bias_v']
bias_h = model['bias_h']
del model_adr,test_adr, model

trainD = train.sum(axis=1)
testD  = test.sum(axis=1)

# compute hidden activations
#train_h = sigmoid((np.dot(train, w_vh) + np.dot(train_lbl, w_sh) + np.outer(trainD, bias_h)))
#test_h = sigmoid(np.dot(test, w_vh) + np.dot(test_lbl, w_sh) +  np.outer(testD, bias_h))
train_h = sigmoid((np.dot(train, w) + np.outer(trainD, bias_h)))
test_h = sigmoid(np.dot(test, w) +  np.outer(testD, bias_h))
#del trainD,testD,w_vh,w_sh,bias_h,bias_v

similarity = np.zeros((test.shape[0], train.shape[0]))
#similarity1 = np.zeros((train.shape[0], test.shape[0]))
for i in range(test_h.shape[0]):
    for j in range(train_h.shape[0]):
        similarity[i,j] = cosine_similarity(test_h[i].reshape(1,-1), train_h[j].reshape(1,-1))
    print "Calculate Cosine Similarity: %d/%d" %(i+1, test.shape[0])
#        similarity1[i,j] = 1 - spatial.distance.cosine(train_h[i].reshape(1,-1), test_h[j].reshape(1,-1))
del i,j

with open('ng/similarity_st/sim','wb') as f:
    pickle.dump(similarity,f,1); f.close(); del train_adr
