import Load

data_adr = '/home/masoud/Desktop/RS/IR/mrmds/dataset_st/test'

data = Load.txt(data_adr)

label = []
for line in data:
    label.append(line.split()[1])

with open('mrmds/test_lbl','w') as f:
    for line in label:    
        f.write('%s\n' %line)
f.close()