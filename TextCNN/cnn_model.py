import tensorflow as tf


class TCNNConfig(object):

	embedding_dim = 64
	num_classes = 10
	seq_length = 600
	num_filters = 256
	kernel_size = 5
	vocab_size = 5000


	hidden_dim = 128
	dropout_keep_prop = 0.5
	learning_rate = 1e-3

	batch_size = 64
	num_epechs = 10

	print_per_batch = 100
	save_per_batch = 10



class TextCNN(object):

	def __init__(self,config):
		self.config = config
		
		#三个待输入的数据
		self.input_x = tf.placeholder(tf.int32,[None, self.config.seq_length],name='input_x')
		self.input_y = tf.placeholder(tf.float32, [None,self.config.num_classes],name='input_y')


		self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

		self.cnn()

	def cnn(self):
		'''CNN模型构建'''
		#词向量映射
		embedding = tf.get_variable('embedding',[self.config.vocab_size,self.config.embedding_dim])
		embedding_inputs = tf.nn.embedding_lookup(embedding,self.input_x)


		with tf.name_scope('cnn'):
			conv = tf.layers.con1d(embedding_inputs,self.config.num_filters,self.config.kernel_size,name='conv')
			gmp = tf.reduce_max(conv,reduction_indices=[1],name='gmp')

		with tf.name_scope('score'):
			fc = tf.layers.dense(gmp,self.config.hidden_dim,name='gmp')
			fc = tf.contrib.layers.dropout(fc,self.keep_prob)
			fc = tf.nn.relu(fc)

			self.logits = tf.layers.dense(fc,self.cong.num,name='fc2')
			self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits),1)


		with tf.name_scope('optimizer'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y)
			self.loss = tf.reduce_mean(cross_entropy)

			self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(cross_entropy)


		with tf.name_scope('accuracy'):
			correct_pred = tf.equal(tf.argmax(self.input_y,1),self.y_pred_cls)
			self.acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
