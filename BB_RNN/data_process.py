#保留一个映射字典
from collections import Counter
import pickle
# my = Counter('df')
# print(my)

# a = Counter(['df','sd','sdf','df'])
# print(a)
# my.update(a)
# print(my)
# print(len(a))
# # print(a.most_common(100000))

# new_a = [x[0] for x in a.most_common(2)]

# new_dict = dict(zip(new_a,range(len(new_a))))
# print(new_dict)
# with open('1.pkl','wb')as f:
# 	pickle.dump(new_dict,f,pickle.HIGHEST_PROTOCOL)


# with open('1.pkl','rb')as f:
# 	my_dict = pickle.load(f)
# print(type(my_dict))
# print(my_dict)





import sys
import tensorflow.contrib.keras as kr

import numpy as np
import tensorflow as tf
path = '../BBdata/'
train_path = path + 'data_224_train.txt'
test_path = path + 'data_224_test.txt'
final_test_path = path + 'test1000.txt'

# vocabulary_size = 10000
# max_length = 600

def build_vocab(vocabulary_size):
	my_dict = Counter()

	i_c = 0
	with open(train_path,encoding='utf-8')as f:
		for line in f:
			if i_c % 1000 ==0:
				print(round(100*i_c/227000,2),'%')
			temp = line.split()
			# print(temp)
			new_temp = temp[:-1]
			temp_dict = Counter(new_temp)
			my_dict.update(temp_dict)
			i_c += 1
			# sys.exit()
	#字典的总长度是：725159
	print('字典的总长度是：{}'.format(len(my_dict)))
	# sys.exit()

	#对总字典做截取，取频次最高的前10000个
	new_dict_list = my_dict.most_common(vocabulary_size)
	new_vocab = [x[0] for x in new_dict_list]
	new_dict = dict(zip(new_vocab,range(len(new_vocab))))

	return new_dict
	# with open('data/my_dict.pkl','wb')as f:
	# 	pickle.dump(new_dict,f,pickle.HIGHEST_PROTOCOL)


def read_vocab():
	with open('data/my_dict.pkl','rb')as f:
		my_dict = pickle.load(f)
		return my_dict

def count_len():
	i_c = 0
	sum_len = 0
	with open(train_path,encoding='utf-8')as f:
		for line in f:
			if i_c % 1000 ==0:
				print(round(100*i_c/227000,2),'%')
			temp = line.split()
			sum_len += len(temp) - 1
			i_c += 1
	print('每个文本分词后的平均长度是：',sum_len/i_c)

def process_file(file_name,my_dict,max_length,categ):
	i_c = 0
	x_data = []
	y_data = []
	with open(file_name,encoding='utf-8')as f:
		for line in f:
			if i_c % 1000 ==0:
				print(round(100*i_c/227000,2),'%')
			temp = line.split()
			temp_x = temp[:-1]

			temp_x_id = [my_dict[x] for x in temp_x if x in my_dict]

			temp_y_id = temp[-1][-1]
			x_data.append(temp_x_id)
			y_data.append(int(temp_y_id))
			i_c += 1
			# if i_c == 2:
			# 	print(len(x_data[1]))
			# 	print(y_data)
			# 	sys.exit()
	#对输入x进行补充和截断处理，保证每一个样本的长度是一样的
	x_pad = kr.preprocessing.sequence.pad_sequences(x_data,padding='pre',truncating='post', maxlen=max_length)
	y_pad = kr.utils.to_categorical(y_data, num_classes=3)
	# np.save('data/x_{}.npy'.format(categ),x_pad)
	# np.save('data/y_{}.npy'.format(categ),y_pad)
	return x_pad, y_pad

def batch_iter(x,y,batch_size=128):
	data_len = len(x)
	num_batch = int((data_len-1)/batch_size) + 1
	#这一步可以用来打乱样本的顺序了，但是我的样本源本来就是已经打乱过的
	# indices = np.random.permutation(np.arange(data_len))

	for i in range(num_batch):
		start_id = i*batch_size
		end_id = min((i+1)*batch_size, data_len)
		yield x[start_id:end_id], y[start_id:end_id]



def build_data(path,vocabulary_size,max_length):
	# my_dict = build_vocab(vocabulary_size)
	# print('字典搭建完成')
	x, y = process_file(path,my_dict,max_length,'none')
	print('x,y数据构建完成')
	return x, y 
# if __name__ == '__main__':
	#计算文本平均长度
	# count_len()
	# 每个文本分词后的平均长度是： 587.2325109649123

	#构建字典
	# build_vocab()

	#读取字典
	# my_dict = read_vocab()
	# print(my_dict)

	#准备训练集和测试集
	# process_file(train_path,my_dict,max_length,'train')
	# process_file(test_path,my_dict,max_length,'test')
	#准备验证集
	# process_file(final_test_path,my_dict,max_length,'1000test')

	#以batchsize来读取数据集
	# x_train = np.load('data/x_train.npy')
	# y_train = np.load('data/y_train.npy')
	# i = 0
	# for a,b in batch_iter(x_train,y_train):
	# 	print(a,b)
	# 	print(a.shape,b.shape)

	# 	if i == 3:
	# 		break
	# 	i += 1