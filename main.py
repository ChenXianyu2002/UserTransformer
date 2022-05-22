import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
a = tf.random_normal(shape=(1, 4, 5))
print(a[:, :, :1:5])
