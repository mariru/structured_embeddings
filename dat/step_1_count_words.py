import glob
import numpy as np
import os
import pandas as pd
import pickle
import re

# Change this to the name of the folder where your dataset is
dataset_name = 'lorem_ipsum'
# Change this to the number of words you want in the vocabulary
V = 300

os.makedirs(dataset_name + '/train/')
os.makedirs(dataset_name + '/valid/')
os.makedirs(dataset_name + '/test/')
files = glob.glob(dataset_name +'/raw/*.txt')
dictionary = {}
count = {}
N = len(files)

for fn in files:
    with open(fn, 'r') as myfile:
        words = re.sub(r'[^a-zA-Z ]',r' ', myfile.read().replace('-\n','').replace('\n',' ')).lower().split()
    data = np.zeros(len(words))
    for idx, word in enumerate(words):
        if word not in dictionary:
            count[len(dictionary)] = 1
            dictionary[word] = len(dictionary)
        data[idx] = dictionary[word]
        count[data[idx]] += 1
    np.save(fn.replace('.txt','.npy'), data.astype('int32'))

df = pd.DataFrame.from_dict(dictionary, orient='index')
cf = pd.DataFrame.from_dict(count, orient='index')
df.columns = ['idx']
cf.columns = ['cnt']
uni = df.join(cf, on = 'idx')

unig = uni.sort_values(by='cnt', ascending = False).reset_index().reset_index()

unig.columns = ['new_idx', 'word', 'old_idx', 'cnt']

old_idx = unig.old_idx.values

files = glob.glob(dataset_name +'/raw/*.npy')
for fname in files:
    dat = np.load(fname)
    new_dat = np.zeros_like(dat) + 2*V
    
    for ni, oi in enumerate(old_idx[:V]):
        new_dat[dat == oi] = ni
    new_dat = new_dat[new_dat < V].astype('int32')
    new_fname = fname.replace('raw/','train/')
    np.save(new_fname, new_dat)

unig.head(V).to_csv(dataset_name + '/unigram.txt',header=False, index = False, sep = '\t', columns = ['word', 'new_idx', 'cnt'])

