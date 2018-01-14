import auxiliary
import random
random.seed(5)

books_train_adr = '/home/paml/Desktop/Masoud_Fatemi/mrmds_10000/sent/books_train'
books_test_adr  = '/home/paml/Desktop/Masoud_Fatemi/mrmds_10000/sent/books_test'
dvd_train_adr = '/home/paml/Desktop/Masoud_Fatemi/mrmds_10000/sent/dvd_train'
dvd_test_adr  = '/home/paml/Desktop/Masoud_Fatemi/mrmds_10000/sent/dvd_test'
electronics_train_adr = '/home/paml/Desktop/Masoud_Fatemi/mrmds_10000/sent/electronics_train'
electronics_test_adr  = '/home/paml/Desktop/Masoud_Fatemi/mrmds_10000/sent/electronics_test'
kitchen_train_adr = '/home/paml/Desktop/Masoud_Fatemi/mrmds_10000/sent/kitchen_train'
kitchen_test_adr  = '/home/paml/Desktop/Masoud_Fatemi/mrmds_10000/sent/kitchen_test'
mr_train_adr = '/home/paml/Desktop/Masoud_Fatemi/mrmds_10000/sent/mr_train'
mr_test_adr  = '/home/paml/Desktop/Masoud_Fatemi/mrmds_10000/sent/mr_test'

books_train = auxiliary.txt(books_train_adr)
books_test = auxiliary.txt(books_test_adr)
dvd_train = auxiliary.txt(dvd_train_adr)
dvd_test = auxiliary.txt(dvd_test_adr)
electronics_train = auxiliary.txt(electronics_train_adr)
electronics_test = auxiliary.txt(electronics_test_adr)
kitchen_train = auxiliary.txt(kitchen_train_adr)
kitchen_test = auxiliary.txt(kitchen_test_adr)
mr_train = auxiliary.txt(mr_train_adr)
mr_test = auxiliary.txt(mr_test_adr)

data = [books_train, books_test,
        dvd_train, dvd_test,
        electronics_train, electronics_test,
        kitchen_train, kitchen_test,
        mr_train, mr_test
        ]
        
positive = []
negative = []
for d in data:
    pos, neg = [], []
    for line in d:
        tokens = line.split()
        if int(tokens[0]) == 1:
            pos.append(line)
        else:
            neg.append(line)
    positive.append(pos)
    negative.append(neg)
       
all_pos = []
for i in range(0,len(positive),2):
    all_pos.append(positive[i] + positive[i+1])
    
all_neg = []
for i in range(0,len(negative),2):
    all_neg.append(negative[i] + negative[i+1])
    
train = []
for d in all_pos:
    for i in range(750):
        train.append(d[i])
for d in all_neg:
    for i in range(750):
        train.append(d[i])
random.shuffle(train)
    
test = []
for d in all_pos:
    for i in range(750,1000):
        test.append(d[i])
for d in all_neg:
    for i in range(750,1000):
        test.append(d[i])
random.shuffle(test)

with open('/home/paml/Desktop/Masoud_Fatemi/RS/IR/mrmds_10000/dataset_t/train', 'w') as f:
    for line in train:
        f.write('%s\n' %line)
f.close()

with open('/home/paml/Desktop/Masoud_Fatemi/RS/IR/mrmds_10000/dataset_t/test', 'w') as f:
    for line in test:
        f.write('%s\n' %line)
f.close()