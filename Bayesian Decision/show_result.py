import numpy as np  
import matplotlib.pyplot as plt

linear_result = open('./Data/result_l.txt')
quadrastic_result = open('./Data/result_q.txt')

linear_result = linear_result.readlines()
quadrastic_result = quadrastic_result.readlines()

linear_cls_1 = []
linear_cls_2 = []
linear_ave = []
linear_cls_1_val = []
linear_cls_2_val = []
linear_ave_val = []

stride_linear = range(32,1025,32)
stride_quadrastic = range(32,1025,32)

quadrastic_cls_1 = []
quadrastic_cls_2 = []
quadrastic_ave = []
quadrastic_cls_1_val = []
quadrastic_cls_2_val = []
quadrastic_ave_val = []

for line in linear_result:
	if line.find("Class 1") != -1 and line.find("TEST") != -1:
		line = line[:-3]
		linear_cls_1.append(line.split(':')[-1])
	elif line.find("Class 2") != -1 and line.find("TEST") != -1:	
		line = line[:-3]
		linear_cls_2.append(line.split(':')[-1])
	elif line.find("Mean ACC") != -1 and line.find("TEST") != -1:
		line = line[:-3]
		linear_ave.append(line.split(':')[-1])
	else:
		continue
		
for line in linear_result:
	if line.find("Class 1") != -1 and line.find("VAL") != -1:
		line = line[:-3]
		linear_cls_1_val.append(line.split(':')[-1])
	elif line.find("Class 2") != -1 and line.find("VAL") != -1:	
		line = line[:-3]
		linear_cls_2_val.append(line.split(':')[-1])
	elif line.find("Mean ACC") != -1 and line.find("VAL") != -1:
		line = line[:-3]
		linear_ave_val.append(line.split(':')[-1])
	else:
		continue

for line in quadrastic_result:
	if line.find("Class 1") != -1 and line.find("TEST") != -1:
		line = line[:-3]
		quadrastic_cls_1.append(line.split(':')[-1])
	elif line.find("Class 2") != -1 and line.find("TEST") != -1:	
		line = line[:-3]
		quadrastic_cls_2.append(line.split(':')[-1])
	elif line.find("Mean ACC") != -1 and line.find("TEST") != -1:
		line = line[:-3]
		quadrastic_ave.append(line.split(':')[-1])
	else:
		continue

for line in quadrastic_result:
	if line.find("Class 1") != -1 and line.find("VAL") != -1:
		line = line[:-3]
		quadrastic_cls_1_val.append(line.split(':')[-1])
	elif line.find("Class 2") != -1 and line.find("VAL") != -1:	
		line = line[:-3]
		quadrastic_cls_2_val.append(line.split(':')[-1])
	elif line.find("Mean ACC") != -1 and line.find("VAL") != -1:
		line = line[:-3]
		quadrastic_ave_val.append(line.split(':')[-1])
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

#print stride_linear

plt.figure(figsize=(8,5))
plt.title("LFD PCA Dim Selection")
plt.xlabel("dim")
plt.ylabel("ACC")
plt.plot(stride_linear, linear_cls_1,'-',color='b',marker='o',label="TEST CLS 1")
plt.plot(stride_linear, linear_cls_2,'-',color='r',marker='o',label="TEST CLS 2")
plt.plot(stride_linear, linear_ave,'-',color='g',marker='o',label="TEST AVE")
plt.plot(stride_linear, linear_cls_1_val,'--',color='b',label="VAL CLS 1")
plt.plot(stride_linear, linear_cls_2_val,'--',color='r',label="VAL CLS 2")
plt.plot(stride_linear, linear_ave_val,'--',color='g',label="VAL AVE")
plt.legend()
plt.grid()
#plt.show()

plt.figure(figsize=(8,5))
plt.title("QFD PCA Dim Selection")
plt.xlabel("dim")
plt.ylabel("ACC")
plt.plot(stride_quadrastic, quadrastic_cls_1,'-',color='b',marker='o',label="TEST CLS 1")
plt.plot(stride_quadrastic, quadrastic_cls_2,'-',color='r',marker='o',label="TEST CLS 2")
plt.plot(stride_quadrastic, quadrastic_ave,'-',color='g',marker='o',label="TEST AVE")
plt.plot(stride_quadrastic, quadrastic_cls_1_val,'--',color='b',label="VAL CLS 1")
plt.plot(stride_quadrastic, quadrastic_cls_2_val,'--',color='r',label="VAL CLS 2")
plt.plot(stride_quadrastic, quadrastic_ave_val,'--',color='g',label="VAL AVE")
plt.legend()
plt.grid()
plt.show()

