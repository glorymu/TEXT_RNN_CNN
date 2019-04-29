import tensorflow as tf
import numpy as np
#c = np.random.random([10, 1])  # 随机生成一个10*1的数组
#b = tf.nn.embedding_lookup(c, [1, 3])#查找数组中的序号为1和3的
# p=tf.Variable(tf.random_normal([10,2]))#生成10*1的张量
# b = tf.nn.embedding_lookup(p, [[1, 3],[2,4]])#查找张量中的序号为1和3的
 
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(b))
#     #print(c)
#     print(sess.run(p))
#     print(p)
#     print(type(p))



a = np.array([[[3,4,15]],
			  [[4,5,6]],
			  [[7,8,9]],
			  [[8,9,10]]])

print(a.shape)
b = tf.Variable(a)
b_ = tf.reshape(b,[2,2,3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b_))
    print(sess.run(tf.reduce_max(b_,reduction_indices=[1])))
