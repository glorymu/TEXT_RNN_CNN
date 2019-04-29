
import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics
 
from BB_rnn import TRNNConfig, TextRNN
from data_process import batch_iter, build_vocab, process_file



def feed_data(x_batch, y_batch, keep_prob):
	feed_dict = {
		model.input_x: x_batch,
		model.input_y: y_batch,
		model.keep_prob: keep_prob
	}
	return feed_dict


def evaluate(sess, x_, y_):
	"""评估在某一数据上的准确率和损失"""
	data_len = len(x_)
	batch_eval = batch_iter(x_, y_, 128)
	total_loss = 0.0
	total_acc = 0.0
	for x_batch, y_batch in batch_eval:
		batch_len = len(x_batch)
		feed_dict = feed_data(x_batch, y_batch, 1.0)
		loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
		total_loss += loss * batch_len
		total_acc += acc * batch_len

	return total_loss / data_len, total_acc / data_len


def get_time_dif(start_time):
	"""获取已使用时间"""
	end_time = time.time()
	time_dif = end_time - start_time
	return timedelta(seconds=int(round(time_dif)))


def train():
	#准备测试集和验证集数据
	global path_train,path_val,my_dict,max_length
	x_train, y_train = process_file(path_train,my_dict,max_length,'none')
	x_val, y_val = process_file(path_val,my_dict,max_length,'none')
	print('数据准备完成')
	# global x_train,x_val,y_train,y_val
	saver = tf.train.Saver()
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	print("Loading training and validation data...")
	start_time = time.time()
	# x_train = np.load('data/x_train.npy')
	# y_train = np.load('data/y_train.npy')
	# x_val = np.load('data/x_test.npy')
	# y_val = np.load('data/y_test.npy')
	# x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
	# x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
	time_dif = get_time_dif(start_time)
	print("Time usage:", time_dif)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	print('Training and evaluating...')
	start_time = time.time()
	total_batch = 0  # 总批次
	last_improved = 0  #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   记录上一次提升批次
	best_acc_val = 0.0
	require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

	flag = False
	for epoch in range(config.num_epochs):
		print('Epoch:', epoch + 1)
		batch_train = batch_iter(x_train, y_train, config.batch_size)
		for x_batch, y_batch in batch_train:
			feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

			if total_batch % config.print_per_batch == 0:
				feed_dict[model.keep_prob] = 1.0
				loss_train, acc_train = sess.run([model.loss, model.acc], feed_dict=feed_dict)
				loss_val, acc_val = evaluate(sess, x_val, y_val)

				if acc_val > best_acc_val:
					# 保存最好结果
					best_acc_val = acc_val
					last_improved = total_batch
					saver.save(sess=sess, save_path=save_path+'_'+str(max_length)+'_'+str(vocab_size))


				time_dif = get_time_dif(start_time)
				msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
					  + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5}'
				print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif,))

			sess.run(model.optim, feed_dict=feed_dict)
			total_batch += 1

def test():
	# x_test = np.load('data/x_1000test.npy')
	# y_test = np.load('data/y_1000test.npy')
	global path_test,my_dict,max_length
	x_test, y_test = process_file(path_test,my_dict,max_length,'none')

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,save_path='checkpoints/best_validation')
		# saver.restore(sess,save_path=save_path)
		loss_test, acc_test = evaluate(sess,x_test,y_test)
		msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
		print(msg.format(loss_test, acc_test))
		batch_size = 128
		data_len = len(x_test)
		num_batch = int((data_len - 1) / batch_size) + 1

		y_test_cls = np.argmax(y_test, 1)
		y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
		for i in range(num_batch):  # 逐批次处理
			start_id = i * batch_size
			end_id = min((i + 1) * batch_size, data_len)
			feed_dict = {
				model.input_x: x_test[start_id:end_id],
				model.keep_prob: 1.0
			}
			y_pred_cls[start_id:end_id] = sess.run(model.y_pre_cls, feed_dict=feed_dict)

		# 评估
		print("Precision, Recall and F1-Score...")
		print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

		# labels = []
		# labels.append(y_test_cls)
		# labels.append(y_pred_cls)
		# np.save('visual/test1000_labels.npy',labels)
		# 混淆矩阵
		# print("Confusion Matrix...")
		# cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
		# print(cm)

		# time_dif = get_time_dif(start_time)
		# print("Time usage:", time_dif)
if __name__ == '__main__':
	#下面是重要参数
	max_length = 500
	vocab_size = 10000
	path_train = '../BBdata/data_224_train.txt' 
	path_val = '../BBdata/data_224_test.txt'
	path_test = '../BBdata/test3000.txt'

	save_dir = 'checkpoints/'
	save_path = save_dir + 'best_validation_{}_{}'.format(str(max_length),str(vocab_size))
	
	my_dict = build_vocab(vocab_size)
	print('字典搭建完成')




	config = TRNNConfig()

	#修正参数
	config.seq_length = max_length
	config.vocab_size = vocab_size


	categories =['货物','工程','服务']
	# if not os.path.exists(vocab_dir):
	# 	build_vocab(train_dir, vocab_dir, config.vocab_size)
	# categories, cat_to_id = read_category()
	# words, word_to_id = read_vocab(vocab_dir)
	# config.vocab_size = len(words)
	model = TextRNN(config)

	# train()
	test()