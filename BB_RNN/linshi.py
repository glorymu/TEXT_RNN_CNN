import tensorflow as tf
import numpy as np
import tensorflow.contrib.keras as kr
# pad_sequence = tf.contrib.keras.preprocessing.sequence.pad_sequences

# a=[[1,2,3],[4,5,6,7]]
# b_len=np.array([len(_) for _ in a])
# bs_packed = pad_sequence(a,maxlen=3,padding='pre',truncating='post',value = 0)

# print(bs_packed)
# print(type(bs_packed))

y_data = np.load('data/y_test.npy')
# print(y_data.shape)
new_data = kr.utils.to_categorical(y_data)
# print(new_data.shape)
# print(y_data[:15])
# print(new_data[:15])
np.save('data/y_test.npy',new_data)
# print(x_data.shape)
# print(x_data[:5])