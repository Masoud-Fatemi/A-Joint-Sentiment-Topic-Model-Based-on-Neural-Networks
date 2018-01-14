import auxiliary

data_adr = 'dataset/originals_2000/mr_test'
data = auxiliary.txt(data_adr)


#detect data set type for assign label
if data_adr.split('/')[-1].split('_')[0] == 'books':
    topic_label = 1
elif data_adr.split('/')[-1].split('_')[0] == 'dvd':
    topic_label = 2
elif data_adr.split('/')[-1].split('_')[0] == 'electronics':
    topic_label = 3
elif data_adr.split('/')[-1].split('_')[0] == 'kitchen':
    topic_label = 4
elif data_adr.split('/')[-1].split('_')[0] == 'mr':
    topic_label = 5

#assign topic label
sent_topic = []
for line in data:
    tmp = []
    tokens = line.split()
    tmp.append(tokens[0])
    tmp.append(topic_label)
    tokens = tokens[1:]
    for token in tokens:
        tmp.append(token)
    sent_topic.append(' '.join(str(i) for i in tmp))

with open('dataset/mr_test', 'w') as f:
    for line in sent_topic:
        f.write('%s\n' %line)
f.close()