import pandas as pd
from mlxtend.data import loadlocal_mnist
import numpy as np
from dataloader import DataLoader
from fullyconnect import MLP2
from val import validate
import matplotlib
from matplotlib import pyplot as plt

#/home/Disk1/konglingjie/shenjingwangluo/hw1/data1/

mypath = '/home/Disk1/konglingjie/shenjingwangluo/hw1/data1/'

train_images,train_labels = loadlocal_mnist(
    images_path=mypath+"train-images.idx3-ubyte",
    labels_path=mypath+"train-labels.idx1-ubyte")
test_images ,test_labels= loadlocal_mnist(
    images_path=mypath+"t10k-images.idx3-ubyte",
    labels_path=mypath+"t10k-labels.idx1-ubyte")




def normalize(x):
    miu = np.mean(x,axis=0,keepdims=True)
    var = np.var(x,axis=0,keepdims=True)
    return (x-miu)/np.sqrt(var+1e-05)
train_images = normalize(train_images)
test_images= normalize(test_images)

input_size=784
hidden_list=[256,128]
output_size=10
lr_list=[0.1]   
l2_list =[1e-04,5e-04]
epochs=1
batch_size=64
result=[]

for hidden in hidden_list:
    for lr in lr_list:
        for l2 in l2_list:
            print(' ')
            print('num_hidden:{},lr:{},l2:{}'.format(hidden, lr, l2))

            train_iter=DataLoader(train_images,train_labels,batch_size)
            test_iter=DataLoader(test_images,test_labels,batch_size)
            mlp=MLP2(input_size,hidden,output_size,lr,l2)
            #mlp.init_model()
            train_loss=[]
            test_loss=[]
            accuracy=[]

            for iteration,data in enumerate(train_iter):
                iteration+=1
                X,y=data
                pro_y=mlp(X)
                loss=mlp.softmax.get_loss(y)
                mlp.backward()
                mlp.update(lr)

                train_loss.append(loss.item())
                acc,test_l=validate(test_iter,mlp)
                test_loss.append(test_l.item())
                accuracy.append(acc.item())
                mlp.lr_decay(iteration)

                if iteration%100==0:
                    print('iteration:{},loss:{},accuracy:{}'.format(iteration, test_l, acc))

            mlp.save_model('/home/Disk1/konglingjie/shenjingwangluo/hw1/klj_hw1/code_hw1/parameters/mlp2-{}hidden-{}lr-{}l2.npy'.format(hidden,lr,l2))

            num=list(range(len(train_loss)))
            plt.figure(figsize=(6, 6), dpi=100)
            plt.plot(num, train_loss, label='trainset loss')
            plt.plot(num, test_loss, label='testset loss')
            plt.xlabel('iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(
                '/home/Disk1/konglingjie/shenjingwangluo/hw1/klj_hw1/code_hw1/picture/loss_{}_{}_{}.png'.format(hidden,lr,l2))

            plt.figure(figsize=(6, 6), dpi=100)
            plt.plot(num, accuracy)
            plt.xlabel('iteration')
            plt.ylabel('testset accuracy')
            plt.savefig(
                '/home/Disk1/konglingjie/shenjingwangluo/hw1/klj_hw1/code_hw1/picture/acc_{}_{}_{}.png'.format(hidden,lr,l2))

            result.append([hidden, lr, l2, acc])

        df = pd.DataFrame(result, columns=['hidden', 'lr', 'l2', 'accuracy'])
        df.to_csv('/home/Disk1/konglingjie/shenjingwangluo/hw1/klj_hw1/code_hw1/result.csv', index=False)




