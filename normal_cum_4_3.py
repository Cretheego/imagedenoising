# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:33:44 2019

@author: Administrator
"""

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
引入残差单元，采用乘子罚函数迭代法
"""
#ÏÔ´æ²»¹»£¬Òª²ÉÓÃµü´úÆ÷
from keras import Input
from keras import layers
import numpy as np
from keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,\
    BatchNormalization,Flatten,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D, \
    UpSampling2D, Conv2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os 
import glob 
import tensorflow as tf  
from keras import backend as K
import random  as rd
from keras.initializers import glorot_uniform
from PIL import Image 
K.set_image_data_format("channels_last")
K.set_learning_phase(1)
#恒等模块——identity_block
def identity_block(X,f,filters,pool=False,up=False):
    #过滤器
    F1,F2 = filters
    X_shortcut = X
    if pool:
        X = MaxPooling2D((2, 2), padding='same')(X)
        X_shortcut = Conv2D(filters=F1,kernel_size=(1,1),strides=(2,2),padding="same",
               kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    if up:
         X_shortcut = Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="same",
               kernel_initializer=glorot_uniform(seed=0))(X)
    #主路径第一部分
    X = BatchNormalization()(X)
    X = Conv2D(filters=F1,kernel_size=(3,3),strides=(1,1),padding="same",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation("relu")(X)
 
    # 主路径第二部分
    X = BatchNormalization()(X)
    X = Conv2D(filters=F2,kernel_size=(3,3),strides=(1,1),padding="same",
              kernel_initializer=glorot_uniform(seed=0))(X)
   
    X_shortcut = BatchNormalization()(X_shortcut)  
    X = layers.add([X,X_shortcut])
    X = Activation("relu")(X)
    
    return X

def cum_coef(y_true, y_pred):
    distance = y_pred - y_true
    mean = K.mean(distance)
    distance -= mean
    Rx = K.mean(K.square(distance))
    global n 
    global cum4_d
    cum_w = K.mean(((distance * distance) * distance) * distance) - 3 * K.square(Rx)
    if n:
        cum4_d = cum_w
        n = n - 1
    return K.mean(K.square(y_pred - y_true),axis=-1),cum_w

def cum_loss(y_true, y_pred):
    normal_loss,cum_w = cum_coef(y_true, y_pred)
    global cum4_d 
    def cum(cum_w):
        global cum4_d        
        print("K.square(cum_w)",K.square(cum_w))
        #with sess.as_default()
        with tf.Session() as sess:
            if K.square(cum_w).eval(session=sess) <= 1.0/4 * K.square(cum4_d).eval(session=sess):
                if coef[1] - coef[0] * cum_w < 0:
                    coef[1] = 0
                else:
                   coef[1] = coef[1] - coef[0] * cum_w 
            else:
                coef[0] = 10 * coef[0]
            cum4_d = cum_w
    return normal_loss - coef[1] * cum4_d + 0.5 * coef[0] * K.square(cum4_d)

def model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy,loss_fun='mean_squared_error'):
    # ¶¨Òåencoder
    input_img = Input(shape=(240, 240, 1))  # (?, 240, 240, 1)
    net = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)  # (?, 240, 240, 16)
    net = Conv2D(32, (3, 3), activation='relu', padding='same')(net)  # (?, 240, 240, 32)
    net = identity_block(net,3,[32,32],pool=True)  # (?, 120, 120, 64)
    #net = identity_block(net,3,[32,32])  # (?, 24, 24, 64)
    encoded = identity_block(net,3,[64,64],pool=True)   # (?, 12, 12, 128)

    net = UpSampling2D((2, 2))(encoded)
    net = identity_block(net,3,[64,64])  # (?, 24, 24, 64)
    net = Conv2D(32, (3, 3), activation='relu', padding='same')(net) 
    net = UpSampling2D((2, 2))(net)
    net = identity_block(net,3,[32,32],up=True) # (?, 48, 48, 32)   
    #net = Conv2D(32, (3, 3), activation='relu', padding='same')(net) 
    #net = Conv2D(32, (3, 3), activation='relu', padding='same')(net) 
    net = Conv2D(16, (3, 3), activation='relu', padding='same')(net)  # (?, 240, 240, 16)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(net)  # (?, 48, 48, 1)

    # Ñ¡¶¨Ä£ÐÍµÄÊäÈë£¬decoded£¨¼´Êä³ö£©µÄ¸ñÊ½
    auto_encoder = Model(input_img, decoded)
    

    auto_encoder.compile(optimizer='adam', loss=loss_fun)
    
    datagen = ImageDataGenerator(zca_whitening=True)
	
    validation_generator = datagen.flow(x_val_noisy, x_val, batch_size=32)

    auto_encoder.fit_generator(datagen.flow(x_train_noisy, x_train, batch_size=32),
                    steps_per_epoch=len(x_train_noisy) / 32, epochs=20,                    
                    validation_data=validation_generator,
                    validation_steps=len(x_val_noisy) / 32)
					
    decoded_imgs = auto_encoder.predict(x_test_noisy)  # ²âÊÔ¼¯ºÏÊäÈë²é¿´Æ÷È¥ÔëÖ®ºóÊä³ö¡£
    return decoded_imgs


#¶ÁÈ¡ÔëÉùÍ¼Ïñ
def trainset(Datapath,n):
    num = len(glob.glob(Datapath+'*.png'))
    print(num)
    imgs = np.zeros((num, 240, 240))
    i = 0
    for imageFile in glob.glob(Datapath+'*.png'):
        #print(imageFile)
        img = np.array(Image.open(imageFile))
        imgs[i] = img
        i += 1
    print(i)
    #print(imgs,np.mean(imgs),np.median(imgs))
    imgs = np.reshape(imgs/255.0,(num,240,240,1))
    #imgs -= np.mean(imgs)    
    return imgs

def immagedenoise(path,filename,n,decoded_imgs):
    isExists = os.path.exists(path)
    if not isExists:
         os.makedirs(path)
    for i in range(n):
        #print((decoded_imgs[i]))
        #out = Image.fromarray(np.uint8(decoded_imgs[i].reshape(240, 240)*255))
        #out.save(path+str(i+1)+'.png','PNG')   
        plt.imsave(path+str(i+1)+'.png',decoded_imgs[i].reshape(240, 240)*255.0)	

		
if __name__ == '__main__':
	coef = [400,5]	
	sigma = 30			
	global cum4_d    
	global n    
	n = 1			
	dataset_clean = 'D:/jupyter_notebook/lena/clean/val/' 
	dataset_noisy = 'D:/jupyter_notebook/lena/noisy/gauss/val/'+'noisy_val'+str(sigma+0.3)+'/'
	x_val = trainset(dataset_clean,3)
	x_val_noisy = trainset(dataset_noisy,3)
	
	dataset_clean = 'D:/jupyter_notebook/lena/clean/train/' 
	dataset_noisy = 'D:/jupyter_notebook/lena/noisy/gauss/train/'+'noisy_train'+str(sigma+0.3)+'/'
	x_train = trainset(dataset_clean,3)
	x_train_noisy = trainset(dataset_noisy,3)
    
	#dataset_noisy = 'D:/jupyter_notebook/lena/split/'
	dataset_clean = 'D:/jupyter_notebook/lena/clean/test/'     
	dataset_noisy = 'D:/jupyter_notebook/lena/noisy/gauss/test/'+'noisy_test'+str(sigma+0.3)+'/'
	x_test = trainset(dataset_clean,3)
	x_test_noisy = trainset(dataset_noisy,3)
	
	print("====",np.shape(x_train))
	print("====",np.shape(x_train_noisy))
	print("====",np.shape(x_test_noisy))
    

	#loss_fun = cum_loss()
	    
	decoded_imgs_cum = model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy,loss_fun = cum_loss)
	print("#####",np.shape(decoded_imgs_cum))

	num = len(x_test_noisy)
	plt.figure(figsize=(18, 4))    
	n = 4
	rdselect = rd.sample(range(num), n)
	print("n====",rdselect)
	for i in range(n):
		# display original images
		ax = plt.subplot(3, n, i + 1)
		plt.imshow(x_test[rdselect[i]].reshape(240, 240))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display noise images
		ax = plt.subplot(3, n, i + 1 + n)
		plt.imshow(decoded_imgs_cum[rdselect[i]].reshape(240, 240))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
        
		ax = plt.subplot(3, n, i + 1 + 2*n)
		plt.imshow(x_test_noisy[rdselect[i]].reshape(240, 240))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)        
	plt.show()
	
	path = 'D:\\jupyter_notebook\\lena\\denoise\\denoise_cum'+str(sigma+0.1053)+'\\'
	immagedenoise(path,'denoise',num,decoded_imgs_cum)
	#print(decoded_imgs_cum[0])
	print(num)
