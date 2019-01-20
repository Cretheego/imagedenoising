# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:40:04 2018

@author: xq
"""

from keras import Input
import numpy as np
from keras.layers import MaxPooling2D, UpSampling2D, Conv2D
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import time 
import os 
import glob 
from keras import backend as K
import random  as rd
from PIL import Image 
import scipy.io as io

def cum_loss(y_true,y_pred):
    distance = y_pred - y_true
    mean = K.mean(distance)  	
    cum3_d = K.mean((distance * distance) * distance)
    cum4_d = K.mean(((distance * distance) * distance) * distance) - 3 * K.mean(distance * distance)
    #return K.mean(K.square(y_pred - y_true),axis=-1) +  20 * K.abs(cum3_d) + 5 * K.abs(cum4_d)
    return K.mean(K.square(y_pred - y_true),axis=-1) + 10 * K.abs(cum3_d) + 5 * K.abs(cum4_d) + 10 * K.square(mean)
    #return K.mean(K.square(y_pred - y_true),axis=-1) +  80 * K.square(cum3_d) + 40 * K.square(cum4_d)

def model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy,loss_fun='mean_squared_error'):
    # 定义encoder
    input_img = Input(shape=(28, 28, 1))  # (?, 28, 28, 1)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # (?, 28, 28, 32)
    x = MaxPooling2D((2, 2), padding='same')(x)  # (?, 14, 14, 32)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # (?, 14, 14, 32)
    encoded = MaxPooling2D((2, 2), padding='same')(x)  # (?, 7, 7, 32)

    # 定义decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)  # (?, 7, 7, 32)
    x = UpSampling2D((2, 2))(x)  # (?, 14, 14, 32)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # (?, 14, 14, 32)
    x = UpSampling2D((2, 2))(x)  # (?, 28, 28, 32)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # (?, 28, 28, 1)

    # 选定模型的输入，decoded（即输出）的格式
    auto_encoder = Model(input_img, decoded)
    # 定义优化目标和损失函数
    auto_encoder.compile(optimizer='sgd', loss=loss_fun)

    # 训练
    history = auto_encoder.fit(x_train_noisy, x_train,  # 输入输出
                     epochs=20,  # 迭代次数
                     batch_size=64,
                     shuffle=True,
                     validation_data=(x_val_noisy, x_val))  # 验证集

    start = time.clock()					 
    decoded_imgs = auto_encoder.predict(x_test_noisy)  # 测试集合输入查看器去噪之后输出。
    elapsed = (time.clock() - start)
    print("Time used:",elapsed/1000)
    io.savemat('./Loss_normal_20.mat', {'Loss_normal_20': history.history['loss']})	
    io.savemat('./Val_Loss_normal_20.mat', {'Val_Loss_normal_20': history.history['val_loss']})	
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()	
    return decoded_imgs


#读取噪声图像
'''def trainset(path,dataset_clean,dataset_noisy):
    img_clean = []
    img_noisy = []
    #num = len(glob.glob(Datapath+dataset_clean+'*.png'))
    #print(num)
    #imgs = np.zeros((num, 28, 28))
    i = 0
    for fpathe,dirs,fs in os.walk(path + dataset_clean):
        for f in fs:
            #print("====",np.shape(mpimg.imread(os.path.join(fpathe,f))))
            img_clean.append(np.ndarray.tolist(np.reshape(mpimg.imread(os.path.join(fpathe,f)),(28,28,1))))
            img_noisy.append(np.ndarray.tolist(np.reshape(mpimg.imread(os.path.join(path + dataset_noisy,f)),(28,28,1))))
            print(np.shape(img_clean))
            #i += 1
			#print("====",mpimg.imread(os.path.join(fpathe,f)))
            #input()
	        
    return np.array(img_clean), np.array(img_noisy)
'''	
def trainset(path,dataset_clean,dataset_noisy):
    num = len(glob.glob(path+dataset_clean+'*.png'))
    print(num)
    imgs_clean = np.zeros((num, 28, 28))
    imgs_noisy = np.zeros((num, 28, 28))
    i = 0	
    for imageFile in glob.glob(path+dataset_clean+'*.png'):
        #print(imageFile)
        img = np.array(Image.open(imageFile))
        #print(np.shape(img))
        imgs_clean[i] = img
        i += 1
    i = 0		
    for imageFile in glob.glob(path+dataset_noisy+'*.png'):
        img = np.array(Image.open(imageFile))
        #print(np.shape(imgs_noisy))		
        imgs_noisy[i] = img
        i += 1		
    print(i,(np.shape(imgs_clean)))
    imgs_clean = np.reshape(imgs_clean,(num,28,28,1))
    imgs_clean = imgs_clean.astype('float32')/255
    imgs_noisy = np.reshape(imgs_noisy,(num,28,28,1))
    imgs_noisy = imgs_noisy.astype('float32')/255
    #imgs -= np.mean(imgs)    
    return imgs_clean,imgs_noisy

def immagedenoise(path,filename,n,decoded_imgs):
    isExists = os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
    for i in range(n):
        plt.imsave(path+str(i+1)+'.png',decoded_imgs[i].reshape(28, 28))	

sigma = 50
path = 'D:/jupyter_notebook/minist/' 
dataset_clean = 'clean/val/'
dataset_noisy = '/noisy/gauss/val/'+'noisy_val'+str(sigma+0.3)+'/'
x_val,x_val_noisy = trainset(path,dataset_clean,dataset_noisy)

dataset_clean = 'clean/train/'
dataset_noisy = '/noisy/gauss/train/'+'noisy_train'+str(sigma+0.3)+'/'
x_train,x_train_noisy = trainset(path,dataset_clean,dataset_noisy)

dataset_clean = 'clean/test/'
dataset_noisy = '/noisy/gauss/test/'+'noisy_test'+str(sigma+0.3)+'/'
x_test,x_test_noisy = trainset(path,dataset_clean,dataset_noisy)


print("====",np.shape(x_train))
print("====",np.shape(x_train_noisy))
print("====",np.shape(x_test))
print("====",np.shape(x_test_noisy))


# 区间剪切，超过区间会被转成区间极值
#x_train_noisy = np.clip(x_train_noisy, 0., 1.)
#x_test_noisy = np.clip(x_test_noisy, 0., 1.)
n = 10
plt.figure(figsize=(18, 4))
for i in range(n):
    # display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noise images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
input()
#decoded_imgs_cum = model_train(x_train_noisy,x_train,x_test_noisy,x_test,loss_fun=cum_loss)
#np.save('decoded_imgs_cum.npy', decoded_imgs_cum)
decoded_imgs_normal = model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy) 
#np.save('decoded_imgs_normal.npy', decoded_imgs_normal)


num = len(x_test)
plt.figure(figsize=(18, 4))

rdselect = rd.sample(range(num), n)
print("n====",rdselect)
for i in range(n):
    # display original
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(x_test_noisy[rdselect[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(decoded_imgs_normal[rdselect[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #print(i," mean error = ",np.mean(x_test[i]-decoded_imgs_normal[i]),"\n ")
	
	# display reconstruction
    ax = plt.subplot(4, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs_normal[rdselect[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
	
	# display reconstruction
    ax = plt.subplot(4, n, i + 1 + 3*n)
    plt.imshow(x_test[rdselect[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#sigma = 0.57  #末位为5表示abs,为6表示abs+mean
#path = 'D:\\jupyter_notebook\\minist\\denoise\\clean\\'
#immagedenoise(path,'clean',num,x_test)
#chong xin pai lie tu pian 
#path = 'D:\\jupyter_notebook\\minist\\denoise\\noisy_test'+str(sigma+0.1)+'\\'
#immagedenoise(path,'noisy',num,x_test_noisy)
'''
path = 'D:\\jupyter_notebook\\minist\\denoise\\clean_test\\'
immagedenoise(path,'denoise',num,x_test)
'''
#path = 'D:\\jupyter_notebook\\minist\\denoise\\denoise_normal'+str(sigma+0.35)+'\\'
#immagedenoise(path,'denoise',num,decoded_imgs_normal)