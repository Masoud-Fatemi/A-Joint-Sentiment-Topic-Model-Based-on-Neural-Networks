import  auxiliary



data_adr = '/home/masoud/Desktop/datasets/mds/dvd/train_unigram'
dict_adr = '/home/masoud/Desktop/datasets/vocab_ng'

#load data
data = auxiliary.txt(data_adr); del data_adr
dictionary = auxiliary.txt(dict_adr); del dict_adr

#create libsvm file
libsvm = auxiliary.lib(data, dictionary)
libsvm = auxiliary.sort_lib(libsvm)

#save result
with open('dvd_train_2000', 'w') as f:
    for line in libsvm:
        f.write('%s\n' %line)
    f.close()
del line