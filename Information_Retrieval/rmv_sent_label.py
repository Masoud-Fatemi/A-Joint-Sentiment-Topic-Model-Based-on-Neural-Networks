import Load

data_adr = '/home/masoud/Desktop/RS/IR/ng/dataset_st/test'

data = Load.txt(data_adr)

dataset_t = []
for line in data:
    tokens = line.split()
    tokens = tokens[1:] 
    dataset_t.append(' '.join(str(i) for i in tokens))
    
#with open('/home/masoud/Desktop/RS/IR/ng/test_sent_lbl', 'w') as f:
#    for line in dataset_t:
#        f.write('%s\n' %line)
#f.close()