import numpy as np  
import matplotlib.pyplot as plt

linear_result = open('./result_l.txt')
quadrastic_result = open('./result_q.txt')

linear_result = linear_result.readlines()
quadrastic_result = quadrastic_result.readlines()

linear_cls_1 = []
linear_cls_2 = []
linear_ave = []

stride_linear = range(50,1001,50)
stride_quadrastic = range(20,301,20)

quadrastic_cls_1 = []
quadrastic_cls_2 = []
quadrastic_ave = []

for line in linear_result:
	if line.find("Class 1") != -1:
		line = line[:-2]
		linear_cls_1.append(line.split(':')[-1])
	elif line.find("Class 2") != -1:	
		line = line[:-2]
		linear_cls_2.append(line.split(':')[-1])
	elif line.find("Mean ACC") != -1:
		line = line[:-2]
		linear_ave.append(line.split(':')[-1])
	else:
		continue

for line in quadrastic_result:
	if line.find("Class 1") != -1:
		line = line[:-2]
		quadrastic_cls_1.append(line.split(':')[-1])
	elif line.find("Class 2") != -1:	
		line = line[:-2]
		quadrastic_cls_2.append(line.split(':')[-1])
	elif line.find("Mean ACC") != -1:
		line = line[:-2]
		quadrastic_ave.append(line.split(':')[-1])
	else:
		continue

#print len(quadrastic_cls_1)
#print len(quadrastic_cls_2)
#print len(quadrastic_ave)

#print len(linear_cls_1)
#print len(linear_cls_2)
#print len(linear_ave)

#print len(stride_linear)
#print len(stride_quadrastic)

print stride_linear

plt.figure(figsize=(8,3))
plt.title("LFD PCA Dim Selection")
plt.xlabel("dim")
plt.ylabel("ACC")
plt.plot(stride_linear, linear_cls_1,'-',label="CLS 1")
plt.plot(stride_linear, linear_cls_2,'-',color='r',label="CLS 2")
plt.plot(stride_linear, linear_ave,'-',label="AVE")
plt.legend()
#plt.grid()
#plt.show()

plt.figure(figsize=(8,3))
plt.title("QFD PCA Dim Selection")
plt.xlabel("dim")
plt.ylabel("ACC")
plt.plot(stride_quadrastic, quadrastic_cls_1,'-',label="CLS 1")
plt.plot(stride_quadrastic, quadrastic_cls_2,'-',color='r',label="CLS 2")
plt.plot(stride_quadrastic, quadrastic_ave,'-',label="AVE")
plt.legend()
#plt.grid()
plt.show()


