# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:38:29 2017
@author: Masoud Fatemi
@organization: IUT, Pattern Analysis and Machine Learning Group
@summary: Implementation of the Replicated Softmax model,
          as presented by R. Salakhutdinov & G.E. Hinton
          in http://www.mit.edu/~rsalakhu/papers/repsoft.pdf
@version: 1.0
NN_finetune: tune the neural network for the proposed model.

"""
import numpy as np
from Load import parse
from keras.models import Sequential
from keras.layers import Dense
import help_nn
np.random.seed(7)
epoch = 10
train_adr = '/home/masoud/Desktop/datasets/mds/books/train'
test_adr = '/home/masoud/Desktop/datasets/mds/books/test'

def binary(list):
    m = np.ones(len(list))
    for i in range(len(list)):
        if list[i]=='2':
            m[i] = 0
    return m

#acc = []
#for hidden in range(60,100,10):
    
# Read train and test data
train_x, train_y = help_nn.parse(train_adr)
test_x, test_y = help_nn.parse(test_adr)
train_y = help_nn.vectorize(train_y)    
test_y = help_nn.vectorize(test_y)

# Create NN
nn = Sequential()
nn.add(Dense(25, input_dim=train_x.shape[1], activation='tanh'))
nn.add(Dense(2, activation='softmax'))


# Compile model
nn.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Fit or train  the model
nn.fit(train_x, train_y, epochs=epoch, batch_size=25)

# evaluate the model
scores = nn.evaluate(test_x, test_y)
print("\n%s: %.2f%%" % (nn.metrics_names[1], scores[1]*100))
#acc.append(scores[1]*100)

#prediction = nn.predict(test_x)
#prediction = [round(i[0]) for i in prediction]

