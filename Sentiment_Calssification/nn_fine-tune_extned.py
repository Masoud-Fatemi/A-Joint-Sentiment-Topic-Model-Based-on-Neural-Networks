
import numpy as np
import help_nn
import pickle
from keras.models import Sequential
from keras.layers import Dense


hidden = 70
init = False
epoch = 10
BachSize = 5
train_adr = '/home/masoud/Desktop/datasets/mr_v1/mr_24916/train'
test_adr = '/home/masoud/Desktop/datasets/mr_v1/mr_24916/test'
model_adr = '/home/masoud/Desktop/aux/LearnedModel_mr_st_24916/1000/new/model_mr_24916_1000_'+str(hidden)
if hidden%2 == 0:
    active = 'tanh'
else:
    active = 'sigmoid'
        
#Read the model
model = pickle.load(open(model_adr, 'rb'))
bias_h = model['bias_h']
bias_s = model['bias_s']
w_sh = model['w_sh']
w_vh = model['w_vh']; del model, model_adr
w_vh = w_vh * 0.1
w_sh = w_sh * 0.1
bias_h = bias_h * 0.1
bias_s = bias_s * 0.1





# Read train and test data
train_x, train_y = help_nn.parse(train_adr)
test_x, test_y = help_nn.parse(test_adr)
train_y = help_nn.vectorize(train_y)    
test_y = help_nn.vectorize(test_y)



# Create NN
np.random.seed(7)
nn = Sequential()
nn.add(Dense(hidden, input_dim=train_x.shape[1], activation= 'sigmoid'))
nn.add(Dense(2, activation='softmax'))

#set weights
if init:
    nn.set_weights([w_vh, bias_h, w_sh.T, bias_s])

# Compile model
nn.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Fit or train  the model
nn.fit(train_x, train_y, epochs=epoch, batch_size=BachSize)

# evaluate the model
scores = nn.evaluate(test_x, test_y)
print("\n%s: %.2f%%" % (nn.metrics_names[1], scores[1]*100))
#acc.append(scores[1]*100)

#prediction = nn.predict(test_x)
#prediction = [round(i[0]) for i in prediction]

