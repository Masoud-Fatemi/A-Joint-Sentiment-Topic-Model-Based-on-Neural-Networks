
import sys
import Load
import TopicModel
import pickle

# default parameters
hiddens = 50
epochs = 1
iter = 1
rate = 0.001
batch = 5
proto = 1	# binary pickled mode
weightinit = 0.001
momentum = 0.9
data_adr = 'mrmds_2000/dataset_t/train'


# load train data
print 'loading data...',; sys.stdout.flush()
data = Load.parse_t(data_adr)
print 'done.'
print 'number of documents        = %d' % data.shape[0]
print 'number of lexicon          = %d' % data.shape[1]
print 'number of hidden variables = %d' % hiddens
print 'number of learning epochs  = %d' % epochs
print 'number of CD iterations    = %d' % iter
print 'minibatch size             = %d' % batch
print 'learning rate              = %g' % rate

# Training and save the result of training
Replicated_Softmax = TopicModel.RSM()
result = Replicated_Softmax.train(data, hiddens, epochs, iter, rate, weightinit, momentum, batch)
with open('mrmds/LearnedModel_t/model_'+str(epochs), 'wb') as file:
    pickle.dump (result, file, proto)