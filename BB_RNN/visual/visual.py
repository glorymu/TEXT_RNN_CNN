import numpy as np
data = np.load('test1000_labels.npy')
print(data.shape)
real = data[0]
pred = data[1]
a = (real==pred).astype(np.int32)
num = len(a)
# print(a)
class_name = ['货物','工程','服务']

f = open('../../BBdata/test1000.txt',encoding='utf-8')
g = open('visual_1000.txt','w',encoding='utf-8')
j = 0
for i,line in zip(a,f):
	if i==0:
		g.write('正确类别为：{}	'.format(class_name[real[j]]))
		g.write('错误分类为：{}\n'.format(class_name[pred[j]]))
		g.write(line)
	j += 1
f.close()
g.close()