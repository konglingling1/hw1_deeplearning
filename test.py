import numpy as np
from mlxtend.data import loadlocal_mnist
from dataloader import DataLoader
from fullyconnect import MLP2
import os


mypath = '/home/Disk1/konglingjie/shenjingwangluo/hw1/data1/'


test_images ,test_labels= loadlocal_mnist(
    images_path=mypath+"t10k-images.idx3-ubyte",
    labels_path=mypath+"t10k-labels.idx1-ubyte")






input_size=784 #28*28
hidden=256
output_size=10
batch_size=64
l2=0.0001

test_iter=DataLoader(test_images,test_labels,batch_size)
mlp2=MLP2(input_size,hidden,output_size,lr=0,l2=0)
param_dir="/home/Disk1/konglingjie/shenjingwangluo/hw1/klj_hw1/code_hw1/parameters/mlp2-{}hidden-0.1lr-{}l2.npy".format(hidden,l2)
mymodel=np.load(param_dir,allow_pickle=True).item()

print(mymodel)

mlp2.load_model(param_dir)


accuracy=0
for X,y in test_iter:
    pro_y=mlp2(X)
    accuracy+=(np.argmax(pro_y,axis=1)==y).sum()
print('test accuracy:',accuracy/len(test_iter))
