import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time
import numpy as np


def make_dir(name):
    dir_name = os.path.join('fits', name, 'EF_EMB_'+time.strftime("%y_%m_%d_%H_%M_%S"))
    while os.path.isdir(dir_name):
        time.sleep(np.random.randint(10))
        dir_name = os.path.join('fits', name, 'EF_EMB_'+time.strftime("%y_%m_%d_%H_%M_%S"))
    os.makedirs(dir_name)
    return dir_name

def variable_summaries(summary_name, var):
    with tf.name_scope(summary_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

