import copy
import Load

dictionary_adr = '/home/masoud/Desktop/datasets/vocab_ng'
lexicon_adr = '/home/masoud/Desktop/datasets/mpqa'
ntopic = 5
itration = '200'
size = '2000'
hidden = 20
version = 'new'
model_adr = 'LearnedModel_mr_st_2000/200/old/model_mr_2000_200_'+str(hidden)           
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
    w_weight.append((pos/float(100)) - (neg/float(55)))
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
del i

acc = ((2*ntopic) - cnt)/float(2*ntopic)*100
print "%0.2f%%" %acc
