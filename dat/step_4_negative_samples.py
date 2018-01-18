import glob
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

# Change this to the name of the folder where your dataset is
dataset_name = 'lorem_ipsum'

os.makedirs(dataset_name + '/valid/neg')
os.makedirs(dataset_name + '/test/neg')

def negative_samples(split, fpath='', ns = 20):
    files = glob.glob(fpath + split + '/*.npy')
    dat_stats = pickle.load(open(os.path.join(fpath, "dat_stats.pkl"), "rb"))
    N = np.sum(dat_stats['train']).astype('int32')
    df = pd.read_csv(os.path.join(fpath, 'unigram.txt'), delimiter='\t',header=None)
    counts = df[len(df.columns)-1].values
    counts = (1.0 * counts / N) ** (3.0 / 4)
    unigram = counts / N
    size = tf.placeholder(tf.int32)
    unigram_logits = tf.tile(tf.expand_dims(tf.log(tf.constant(unigram)), [0]), [size, 1])
    n_idx = tf.multinomial(unigram_logits, ns)
    sess = tf.Session()
    for fn in files:
        dat = np.load(fn)
        neg_words = sess.run(n_idx, feed_dict = {size: len(dat)})
        np.save(fn.replace(split,split+'/neg'),neg_words.astype('int32'))

negative_samples('/test', dataset_name)
negative_samples('/valid', dataset_name)
