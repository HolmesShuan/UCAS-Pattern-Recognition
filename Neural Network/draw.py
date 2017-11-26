import numpy as np  
import matplotlib.pyplot as plt

import scipy.io as spio

def data_split(x, intervel):
    y = x[0::intervel]
    return y

network_3_3_3_3_dir = './network_3_3_3_3/'
network_3_7_7_3_dir = './network_3_7_7_3/'

lr_type = 'inv'
batchsize = '1'

test_acc_file = 'test_acc_' + lr_type + '_batchsize_' + batchsize + '.mat'
test_loss_file = 'test_loss_' + lr_type + '_batchsize_' + batchsize + '.mat'
train_acc_file = 'train_acc_' + lr_type + '_batchsize_' + batchsize + '.mat'
train_loss_file = 'train_loss_' + lr_type + '_batchsize_' + batchsize + '.mat'

## Network Structure
test_loss_batchsize_1_net_3_3 = spio.loadmat(network_3_3_3_3_dir+test_loss_file)['test_loss']
train_loss_batchsize_1_net_3_3 = spio.loadmat(network_3_3_3_3_dir+train_loss_file)['train_loss']
test_loss_batchsize_1_net_7_7 = spio.loadmat(network_3_7_7_3_dir+test_loss_file)['test_loss']
train_loss_batchsize_1_net_7_7 = spio.loadmat(network_3_7_7_3_dir+train_loss_file)['train_loss']

train_loss_batchsize_1_net_3_3 = data_split(train_loss_batchsize_1_net_3_3, 200);
train_loss_batchsize_1_net_7_7 = data_split(train_loss_batchsize_1_net_7_7, 200);

batchsize = '24'

test_acc_file = 'test_acc_' + lr_type + '_batchsize_' + batchsize + '.mat'
test_loss_file = 'test_loss_' + lr_type + '_batchsize_' + batchsize + '.mat'
train_acc_file = 'train_acc_' + lr_type + '_batchsize_' + batchsize + '.mat'
train_loss_file = 'train_loss_' + lr_type + '_batchsize_' + batchsize + '.mat'

test_loss_batchsize_24_net_3_3 = spio.loadmat(network_3_3_3_3_dir+test_loss_file)['test_loss']
train_loss_batchsize_24_net_3_3 = spio.loadmat(network_3_3_3_3_dir+train_loss_file)['train_loss']
test_loss_batchsize_24_net_7_7 = spio.loadmat(network_3_7_7_3_dir+test_loss_file)['test_loss']
train_loss_batchsize_24_net_7_7 = spio.loadmat(network_3_7_7_3_dir+train_loss_file)['train_loss']

train_loss_batchsize_24_net_3_3 = data_split(train_loss_batchsize_24_net_3_3, 200);
train_loss_batchsize_24_net_7_7 = data_split(train_loss_batchsize_24_net_7_7, 200);

plt.figure()
plt.title("Hidden Nodes Comparsion.")
plt.xlabel("iter")
plt.ylabel("test loss")
plt.plot(test_loss_batchsize_1_net_3_3, '-', label="batchsize=1 structure : 3-3-3-3")
plt.plot(test_loss_batchsize_1_net_7_7, '--',label="batchsize=1 structure : 3-7-7-3")
plt.plot(test_loss_batchsize_24_net_3_3,'-', label="batch=24 3-3-3-3")
plt.plot(test_loss_batchsize_24_net_7_7,'--',label="batch=24 3-7-7-3")
plt.legend()
plt.grid()
plt.show()

## batchsize

batchsize = '13'

test_acc_file = 'test_acc_' + lr_type + '_batchsize_' + batchsize + '.mat'
test_loss_file = 'test_loss_' + lr_type + '_batchsize_' + batchsize + '.mat'
train_acc_file = 'train_acc_' + lr_type + '_batchsize_' + batchsize + '.mat'
train_loss_file = 'train_loss_' + lr_type + '_batchsize_' + batchsize + '.mat'

test_loss_batchsize_13_net_3_3 = spio.loadmat(network_3_3_3_3_dir+test_loss_file)['test_loss']
train_loss_batchsize_13_net_3_3 = spio.loadmat(network_3_3_3_3_dir+train_loss_file)['train_loss']
test_loss_batchsize_13_net_7_7 = spio.loadmat(network_3_7_7_3_dir+test_loss_file)['test_loss']
train_loss_batchsize_13_net_7_7 = spio.loadmat(network_3_7_7_3_dir+train_loss_file)['train_loss']

train_loss_batchsize_13_net_3_3 = data_split(train_loss_batchsize_13_net_3_3, 200);
train_loss_batchsize_13_net_7_7 = data_split(train_loss_batchsize_13_net_7_7, 200);

plt.figure()
plt.title("BatchSize Comparsion.")
plt.xlabel("iter")
plt.ylabel("training loss")
plt.plot(train_loss_batchsize_1_net_7_7, '-', label="batch=1 structure : 3-7-7-3")
plt.plot(train_loss_batchsize_13_net_7_7, '--',label="batch=13 structure : 3-7-7-3")
plt.plot(train_loss_batchsize_24_net_7_7,'-', label="batch=24 structure : 3-7-7-3")
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.title("BatchSize Comparsion.")
plt.xlabel("iter")
plt.ylabel("test loss")
plt.plot(test_loss_batchsize_1_net_7_7, '-', label="batch=1 structure : 3-7-7-3")
plt.plot(test_loss_batchsize_13_net_7_7, '--',label="batch=13 structure : 3-7-7-3")
plt.plot(test_loss_batchsize_24_net_7_7,'-', label="batch=24 structure : 3-7-7-3")
plt.legend()
plt.grid()
plt.show()

## learning rate

lr_type = 'poly'
batchsize = '24'

test_acc_file = 'test_acc_' + lr_type + '_batchsize_' + batchsize + '.mat'
test_loss_file = 'test_loss_' + lr_type + '_batchsize_' + batchsize + '.mat'
train_acc_file = 'train_acc_' + lr_type + '_batchsize_' + batchsize + '.mat'
train_loss_file = 'train_loss_' + lr_type + '_batchsize_' + batchsize + '.mat'

test_loss_batchsize_24_net_3_3_poly = spio.loadmat(network_3_3_3_3_dir+test_loss_file)['test_loss']
train_loss_batchsize_24_net_3_3_poly = spio.loadmat(network_3_3_3_3_dir+train_loss_file)['train_loss']
test_loss_batchsize_24_net_7_7_poly = spio.loadmat(network_3_7_7_3_dir+test_loss_file)['test_loss']
train_loss_batchsize_24_net_7_7_poly = spio.loadmat(network_3_7_7_3_dir+train_loss_file)['train_loss']

test_acc_batchsize_24_net_3_3_poly = spio.loadmat(network_3_3_3_3_dir+test_acc_file)['test_acc']
train_acc_batchsize_24_net_3_3_poly = spio.loadmat(network_3_3_3_3_dir+train_acc_file)['train_acc']
test_acc_batchsize_24_net_7_7_poly = spio.loadmat(network_3_7_7_3_dir+test_acc_file)['test_acc']
train_acc_batchsize_24_net_7_7_poly = spio.loadmat(network_3_7_7_3_dir+train_acc_file)['train_acc']

train_loss_batchsize_24_net_3_3_poly = data_split(train_loss_batchsize_24_net_3_3_poly, 200);
train_loss_batchsize_24_net_7_7_poly = data_split(train_loss_batchsize_24_net_7_7_poly, 200);

train_acc_batchsize_24_net_3_3_poly = data_split(train_acc_batchsize_24_net_3_3_poly, 200);
train_acc_batchsize_24_net_7_7_poly = data_split(train_acc_batchsize_24_net_7_7_poly, 200);

plt.figure()
plt.title("Learning Rate Comparsion.")
plt.xlabel("iter")
plt.ylabel("test loss")
plt.plot(test_loss_batchsize_24_net_3_3_poly, '-', label="poly batch=24 structure : 3-3-3-3")
plt.plot(test_loss_batchsize_24_net_7_7_poly, '--',label="poly batch=24 structure : 3-7-7-3")
plt.plot(test_loss_batchsize_24_net_3_3,'-', label="inv batch=24 structure : 3-3-3-3")
plt.plot(test_loss_batchsize_24_net_7_7, '--',label="inv batch=24 structure : 3-7-7-3")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title("Learning Rate Comparsion.")
plt.xlabel("iter")
plt.ylabel("training loss")
plt.plot(train_loss_batchsize_24_net_3_3_poly, '-', label="poly batch=24 structure : 3-3-3-3")
plt.plot(train_loss_batchsize_24_net_7_7_poly, '--',label="poly batch=24 structure : 3-7-7-3")
plt.plot(train_loss_batchsize_24_net_3_3,'-', label="inv batch=24 structure : 3-3-3-3")
plt.plot(train_loss_batchsize_24_net_7_7, '--',label="inv batch=24 structure : 3-7-7-3")
plt.legend()
plt.grid()
plt.show()

lr_type = 'inv'
batchsize = '24'

test_acc_file = 'test_acc_' + lr_type + '_batchsize_' + batchsize + '.mat'
test_loss_file = 'test_loss_' + lr_type + '_batchsize_' + batchsize + '.mat'
train_acc_file = 'train_acc_' + lr_type + '_batchsize_' + batchsize + '.mat'
train_loss_file = 'train_loss_' + lr_type + '_batchsize_' + batchsize + '.mat'

test_acc_batchsize_24_net_3_3 = spio.loadmat(network_3_3_3_3_dir+test_acc_file)['test_acc']
train_acc_batchsize_24_net_3_3 = spio.loadmat(network_3_3_3_3_dir+train_acc_file)['train_acc']
test_acc_batchsize_24_net_7_7 = spio.loadmat(network_3_7_7_3_dir+test_acc_file)['test_acc']
train_acc_batchsize_24_net_7_7 = spio.loadmat(network_3_7_7_3_dir+train_acc_file)['train_acc']

train_acc_batchsize_24_net_3_3 = data_split(train_acc_batchsize_24_net_3_3, 200);
train_acc_batchsize_24_net_7_7 = data_split(train_acc_batchsize_24_net_7_7, 200);

plt.figure()
plt.title("Learning Rate Comparsion.")
plt.xlabel("iter")
plt.ylabel("training accuracy")
plt.plot(train_acc_batchsize_24_net_3_3_poly, '-', label="poly batch=24 structure : 3-3-3-3")
plt.plot(train_acc_batchsize_24_net_7_7_poly, '--',label="poly batch=24 structure : 3-7-7-3")
plt.plot(train_acc_batchsize_24_net_3_3,'-', label="inv batch=24 structure : 3-3-3-3")
plt.plot(train_acc_batchsize_24_net_7_7, '--',label="inv batch=24 structure : 3-7-7-3")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title("Learning Rate Comparsion.")
plt.xlabel("iter")
plt.ylabel("test accuracy")
plt.plot(test_acc_batchsize_24_net_3_3_poly, '-', label="poly batch=24 structure : 3-3-3-3")
plt.plot(test_acc_batchsize_24_net_7_7_poly, '--',label="poly batch=24 structure : 3-7-7-3")
plt.plot(test_acc_batchsize_24_net_3_3,'-', label="inv batch=24 structure : 3-3-3-3")
plt.plot(test_acc_batchsize_24_net_7_7, '--',label="inv batch=24 structure : 3-7-7-3")
plt.legend()
plt.grid()
plt.show()

