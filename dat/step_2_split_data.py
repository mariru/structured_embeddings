import glob
import numpy as np
import pickle
import pandas as pd

# Change this to the name of the folder where your dataset is
dataset_name = 'lorem_ipsum'

files = glob.glob(dataset_name+'/train/*.npy')

df = pd.read_csv(dataset_name+'/unigram.txt', delimiter='\t',header=None)
cnt = df[2].values
cnt = 1.0*cnt/cnt.sum()

for fname in files:
    dat = np.load(fname)
    prob = np.random.uniform(0,1,dat.shape)
    p = 1 - np.sqrt((10.0**(-5))/cnt[dat])
    dat = dat[prob > p]

    split = int(0.1*len(dat))
    i = np.random.randint(len(dat))

    dat = np.roll(dat, i)
    test_dat = dat[:split]
    dat = dat[split:]

    dat = np.roll(dat, i)
    valid_dat = dat[:split]
    dat = dat[split:]

    np.save(fname.replace('train','test'), test_dat)
    np.save(fname.replace('train','valid'), valid_dat)
    np.save(fname, dat)

