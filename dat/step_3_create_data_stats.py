import glob
import os
import numpy as np
import pickle


# Change this to the name of the folder where your dataset is
dataset_name = 'lorem_ipsum'

# Change this to a list of the groups 
states = ['A', 'B']



# No need to modify any code below
#######################################################
dat_stats={}
dat_stats['name'] = dataset_name
dat_stats['T_bins'] = states
T = len(dat_stats['T_bins'])

def count_words(split):
    dat_stats[split] = np.zeros(T)
    files = glob.glob(dataset_name + '/'+ split + '/*.npy')
    for t, i in enumerate(dat_stats['T_bins']):
        dat_files = [f for f in files if os.path.basename(f).split('_')[0] == i]
        for fname in dat_files:
            dat = np.load(fname)
            dat_stats[split][t] += len(dat)

count_words('train')
count_words('test')
count_words('valid')

pickle.dump(dat_stats, open(dataset_name + '/dat_stats.pkl', "wb" ) )
