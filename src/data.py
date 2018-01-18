import numpy as np
import os
import pandas as pd
import pickle
import glob

class bern_emb_data():
    def __init__(self, cs, ns, fpath, hierarchical, n_epochs=1, debug=False, remove_stopwords=False):
        assert cs%2 == 0
        self.cs = cs
        self.ns = ns
        self.n_epochs = n_epochs
        self.hierarchical = hierarchical
        dat_stats = pickle.load(open(os.path.join(fpath, "dat_stats.pkl"), "rb"))
        self.T = len(dat_stats['T_bins'])
        self.states = dat_stats['T_bins']
        self.name = dat_stats['name']
        if debug:
            print('ONLY USING 100th of DATA')
            self.n_epochs = self.n_epochs/100
            self.eval_states = self.states[:3]
        else:    
            self.eval_states = self.states
        self.N = np.sum(dat_stats['train']).astype('int32')
        self.n_train = np.maximum(dat_stats['train']/n_epochs, 2*np.ones_like(dat_stats['train'])).astype('int32')
        self.n_valid = dat_stats['valid'].astype('int32')
        self.n_test = dat_stats['test'].astype('int32')

	# load vocabulary
	df = pd.read_csv(os.path.join(fpath, 'unigram.txt'), delimiter='\t',header=None)
	self.labels = df[0].values
	self.counts = df[len(df.columns)-1].values
        counts = (1.0 * self.counts / self.N) ** (3.0 / 4)
        self.unigram = counts / self.N
        self.w_idx = range(len(self.labels))
        if remove_stopwords:
	    sw_df = pd.read_csv(os.path.join(fpath, 'stop_words.txt'), delimiter='\t',header=None)
            stop_words = sw_df[0].values 
            self.w_idx = [i for i, w in enumerate(self.labels) if w not in stop_words]
            self.labels = self.labels[self.w_idx]
            self.counts = self.counts[self.w_idx]
            self.unigram = self.unigram[self.w_idx]
        self.L = len(self.labels)
        self.dictionary = dict(zip(self.labels,range(self.L)))

        # data generator (training)
        train_files = glob.glob(os.path.join(fpath,'train','*.npy'))
        if self.hierarchical:
            self.batch = {}
            for t, i in enumerate(self.states):
                print(i)
                print(len([f for f in train_files if os.path.basename(f).split('_')[0] == i]))
                self.batch[i] = self.batch_generator(self.n_train[t] + self.cs, 
                                    [f for f in train_files if os.path.basename(f).split('_')[0] == i])
        else:
            self.batch = self.batch_generator(self.n_train.sum() + self.cs, train_files)

        # data generator (test)
        test_files = glob.glob(os.path.join(fpath,'test','*.npy'))
        self.test_data = {}
        for state in self.states:
            self.test_data[state] = self.data_and_negative_samples(
                                        [f for f in test_files if os.path.basename(f).split('_')[0] == state])

        # data generator (valid)
        valid_files = glob.glob(os.path.join(fpath,'valid','*.npy'))
        self.valid_data = {}
        for state in self.states:
            self.valid_data[state] = self.data_and_negative_samples(
                                         [f for f in valid_files if os.path.basename(f).split('_')[0] == state])

    def load_file(self, fn):
        with open(fn, 'r') as myfile:
            words = myfile.read().replace('\n', '').split()
        data = np.zeros(len(words))
        for idx, word in enumerate(words):
            if word in self.dictionary:
                data[idx] = self.dictionary[word]
        return data

    def batch_generator(self, batch_size, files):
        f_idx = 0
        data = np.load(files[f_idx])
        while True:
            if data.shape[0] < batch_size:
                f_idx+=1
                if (f_idx>=len(files)):
                    f_idx = 0
        	data_new = np.load(files[f_idx])
                data = np.hstack([data, data_new])
                if data.shape[0] < batch_size:
                    continue
            words = data[:batch_size]
            data = data[batch_size:]
            yield words.astype('int32')

    def data_and_negative_samples(self, files):
        data = np.array([]).astype('int32')
        neg_data = np.array(self.ns*[[]]).astype('int32').T
        for fn in files:
            data_new = np.load(fn)
            data = np.hstack([data, data_new])
            neg_data_new = np.load(fn.replace('test','test/neg').replace('valid','valid/neg'))[:,:self.ns]
            neg_data = np.vstack([neg_data, neg_data_new])
        yield (data.astype('int32'), neg_data.astype('int32'))
    
    def train_feed(self, placeholder):
        if self.hierarchical:
            feed_dict = {}
            for state in self.states:
                feed_dict[placeholder[state]] = self.batch[state].next()
            return feed_dict
        else:
            return {placeholder: self.batch.next()}
