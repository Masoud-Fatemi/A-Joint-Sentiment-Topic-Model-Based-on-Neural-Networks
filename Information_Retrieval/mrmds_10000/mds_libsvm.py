import  auxiliary



data_adr = '/home/paml/Desktop/Masoud_Fatemi/mds/electronics/train_unigram'
dict_adr = '/home/paml/Desktop/Masoud_Fatemi/vocab_rcv'

#load data
data = auxiliary.txt(data_adr); del data_adr
dictionary = auxiliary.txt(dict_adr); del dict_adr

#create libsvm file
libsvm = auxiliary.lib(data, dictionary)
libsvm = auxiliary.sort_lib(libsvm)

#save result
with open('/home/paml/Desktop/Masoud_Fatemi/mrmds_10000/sent/electronics_train', 'w') as f:
    for line in libsvm:
        f.write('%s\n' %line)
    f.close()
del line