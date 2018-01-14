
import sys
import Load
import SentTopicModel
import pickle
#import time

# default parameters
hiddens = 50
sentiment = 2
epochs = 500
iter = 1
rate = 0.001
batch = 5
proto = 1	# binary pickled mode
weightinit = 0.001
momentum = 0.9
data_adr = 'mrmds_2000/dataset_st/train'

# load train data
print 'loading data...',; sys.stdout.flush()
data = Load.parse(data_adr)
print 'done.'
print 'number of documents        = %d' % data.shape[0]
print 'number of dictionary       = %d' %(data.shape[1]-1)
print 'number of hiddens          = %d' % hiddens
print 'number of sentiment        = %d' % sentiment
print 'number of learning epochs  = %d' % epochs
print 'number of CD iterations    = %d' % iter
print 'minibatch size             = %d' % batch
print 'learning rate              = %g' % rate

# Training and save the result of training
Sent_Replicated_Softmax = SentTopicModel.RSM()
result = Sent_Replicated_Softmax.train(data, hiddens,
                                       sentiment, epochs, iter, rate, weightinit, momentum, batch)
with open('mrmds/LearnedModel_st/model_'+str(epochs), 'wb') as file:
    pickle.dump (result, file, proto)