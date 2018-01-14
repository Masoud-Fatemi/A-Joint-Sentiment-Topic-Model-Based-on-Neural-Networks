

import numpy as np
np.random.seed(5)

data_adr = '/home/masoud/Desktop/datasets/ng/test'
lex_adr = '/home/masoud/Desktop/datasets/mpqa'
dict_adr = '/home/masoud/Desktop/datasets/vocab_ng'


#load data, dictionary and sentiment lexicon
data = open(data_adr).read().split('\n')
data = [i for i in data if i.rstrip()]
dictionary = open(dict_adr).read().split('\n')
dictionary = [i for i in dictionary if i.rstrip()]
lexicon = open(lex_adr, 'r').read().split('\n')
lexicon = [i for i in lexicon if i.rstrip()]
lexicon = {i.split('\t')[0]:i.split('\t')[1:] for i in lexicon}
del data_adr, lex_adr, i, dict_adr

#extract positive and negative ids from dictionary
common_pos = []
common_neg = []
for vocab in dictionary:
    if vocab in lexicon:
        if lexicon[vocab][1]>lexicon[vocab][2]:
            common_pos.append(dictionary.index(vocab)+1)
        else: 
            common_neg.append(dictionary.index(vocab)+1)
del vocab

#assigne sentiment label to data based on number of sentiment words
sent_lbl = []
data_new = []
for line in data:
    pos, neg = 0, 0
    tokens = line.split()[1:]
    for token in tokens:
        if int(token.split(':')[0]) in common_pos:
            pos += 1
        elif int(token.split(':')[0]) in common_neg:
            neg += 1
    if pos/float(100) > neg/float(55):
        sent_lbl.append(1)
        data_new.append('1 ' + line)
    elif pos/float(100) < neg/float(55):
        sent_lbl.append(2)
        data_new.append('2 ' + line)
    elif pos/float(100) == neg/float(55): 
        sent_lbl.append(int(np.random.choice([1, 2], p = [1./2, 1./2])))
        data_new.append(str(np.random.choice([1, 2], p = [1./2, 1./2]))+' '+ line)
#del pos, neg, token, tokens, line 

with open('/home/masoud/Desktop/datasets/ng/test_st', 'w') as f:
    for line in data_new:
        f.write('%s\n' %line)
f.close()
    
    


            