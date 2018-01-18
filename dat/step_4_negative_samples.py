import glob
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

# Change this to the name of the folder where your dataset is
dataset_name = 'lorem_ipsum'

# Change this to the maximal number of negative samples (per positive sample) you plan to use during training and evaluation. (Note: the negative samples drawn in this step will be used during evaluation)
negative_samples = 20


# No need to modify any code below
#######################################################
os.makedirs(dataset_name + '/valid/neg')
os.makedirs(dataset_name + '/test/neg')

def negative_samples(split, fpath='', ns = negative_samples):
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
