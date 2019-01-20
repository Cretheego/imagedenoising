# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 19:47:04 2019

@author: Administrator
"""

import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,\
    BatchNormalization,Flatten,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D, \
    UpSampling2D, Conv2D, Dropout
from keras.models import Model,load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
import pydoc
from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os 
import glob 
from keras import backend as K
import random  as rd
from PIL import Image 
K.set_image_data_format("channels_last")
K.set_learning_phase(1)

def cum_loss(y_true,y_pred):
    distance = y_pred - y_true
    mean = K.mean(distance)
    distance -= mean
    cum3_d = K.mean((distance * distance) * distance)
    cum4_d = K.mean(((distance * distance) * distance) * distance) - 3 * K.mean(distance * distance)* K.mean(distance * distance)
    #return K.mean(K.square(y_pred - y_true),axis=-1) +  20 * K.abs(cum3_d) + 5 * K.abs(cum4_d)
    return K.mean(K.square(y_pred - y_true),axis=-1) + 5* K.square(cum4_d) #40 *  K.square(cum3_d)  

def trainset(Datapath,n):
    num = len(glob.glob(Datapath+'*.png'))
    print(num)
    imgs = np.zeros((num, 48, 48))
    i = 0
    for imageFile in glob.glob(Datapath+'*.png'):
        #print(imageFile)
        img = np.array(Image.open(imageFile))
        imgs[i] = img
        i += 1
    print(i)
    imgs = np.reshape(imgs/255.0,(num,48,48,1))
    #imgs = imgs.astype('float32')/255
    #imgs = imgs - imgs.mean()    
    return imgs

def immagedenoise(path,filename,n,decoded_imgs):
    isExists = os.path.exists(path)
    if not isExists:
         os.makedirs(path)
    for i in range(n):
        plt.imsave(path+str(i+1)+'.png',decoded_imgs[i].reshape(48, 48))	
        
#恒等模块——identity_block
def identity_block(X,f,filters,pool=False,up=False):
    """
    三层的恒等残差块
    param :
    X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
    f -- 整数，指定主路径的中间 CONV 窗口的形状
    filters -- python整数列表，定义主路径的CONV层中的过滤器数目
    stage -- 整数，用于命名层，取决于它们在网络中的位置
    block --字符串/字符，用于命名层，取决于它们在网络中的位置
    return:
    X -- 三层的恒等残差块的输出，维度为：(n_H, n_W, n_C)
    """
    #过滤器
    F1,F2 = filters
    
    #保存输入值,后面将需要添加回主路径
    X_shortcut = X
    if pool:
        X = MaxPooling2D((2, 2), padding='same')(X)
        X_shortcut = Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="same",
               kernel_initializer=glorot_uniform(seed=0))(X)
    if up:
        X = UpSampling2D((2, 2))(X)
        X_shortcut = Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="same",
               kernel_initializer=glorot_uniform(seed=0))(X)
    #主路径第一部分
    X = BatchNormalization()(X)
    X = Conv2D(filters=F1,kernel_size=(f,f),strides=(1,1),padding="same",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation("relu")(X)
 
    # 主路径第二部分
    X = BatchNormalization()(X)
    X = Conv2D(filters=F2,kernel_size=(3,3),strides=(1,1),padding="same",
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation("relu")(X)
    
    # 主路径第三部分
    #X_shortcut = Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="same",
    #   name=conv_name_base+"2a",kernel_initializer=glorot_uniform(seed=0))(X)
    # 主路径最后部分,为主路径添加shortcut并通过relu激活
    X = layers.add([X,X_shortcut])
    #X = Activation("relu")(X) 
    return X

def model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy,loss_fun='mean_squared_error'):
    # ¶¨Òåencoder
    input_img = Input(shape=(48, 48, 1))  # (?, 50, 50, 1)
    net = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)  # (?, 48, 48, 32)
    net = identity_block(net,3,[64,64])  # (?, 24, 24, 64)
    net = identity_block(net,3,[32,32],pool=True)  # (?, 24, 24, 64)
    net = identity_block(net,3,[32,32])  # (?, 24, 24, 64)
    encoded = identity_block(net,3,[16,16],pool=True)   # (?, 12, 12, 128)
    #encoded = identity_block(net,3,[128,128],pool=True)    # (?, 6, 6, 128)
    
    net = identity_block(encoded,3,[16,16])  # (?, 6, 6, 128)
    net = identity_block(net,3,[32,32],up=True)  # (?, 24, 24, 64)
    net = identity_block(net,3,[32,32])  # (?, 24, 24, 64)    
    net = identity_block(net,3,[64,64],up=True) # (?, 48, 48, 32)
    net = identity_block(net,3,[64,64])  # (?, 24, 24, 64)    
    #net = BatchNormalization()(net)
    #net = Activation("relu")(net)
    #net = Dropout(0.25)(net)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(net)  # (?, 48, 48, 1)

    auto_encoder = Model(input_img, decoded)
    auto_encoder.compile(optimizer='adam', loss=loss_fun)
     
    datagen = ImageDataGenerator(zca_whitening=True)
    validation_generator = datagen.flow(x_val_noisy, x_val, batch_size=32)

    auto_encoder.fit_generator(datagen.flow(x_train_noisy, x_train, batch_size=32),
                    steps_per_epoch=len(x_train_noisy) / 32, epochs=40,                    
                    validation_data=validation_generator,
                    validation_steps=len(x_val_noisy) / 32)
					
    decoded_imgs = auto_encoder.predict(x_test_noisy)  
    return decoded_imgs

if __name__ == '__main__':
	sigma = 20			
	dataset_clean = 'D:/jupyter_notebook/lena/clean48/val/' 
	dataset_noisy = 'D:/jupyter_notebook/lena/noisy48/gauss/val/'+'noisy_val'+str(sigma+0.3)+'/'
	x_val = trainset(dataset_clean,3)
	x_val_noisy = trainset(dataset_noisy,3)
	
	dataset_clean = 'D:/jupyter_notebook/lena/clean48/train/' 
	dataset_noisy = 'D:/jupyter_notebook/lena/noisy48/gauss/train/'+'noisy_train'+str(sigma+0.3)+'/'
	x_train = trainset(dataset_clean,3)
	x_train_noisy = trainset(dataset_noisy,3)
    
	#dataset_noisy = 'D:/jupyter_notebook/lena/split/'
	dataset_clean = 'D:/jupyter_notebook/lena/clean48/test/'     
	dataset_noisy = 'D:/jupyter_notebook/lena/noisy48/gauss/test/'+'noisy_test'+str(sigma+0.3)+'/'
	x_test = trainset(dataset_clean,3)
	x_test_noisy = trainset(dataset_noisy,3)
	
	print("====",np.shape(x_train))
	print("====",np.shape(x_train_noisy))
	print("====",np.shape(x_test_noisy))
	n = 8
    
	decoded_imgs_cum = model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy,loss_fun=cum_loss)
	print("#####",np.shape(decoded_imgs_cum))
	#decoded_imgs = model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy,loss_fun=cum_loss1)

	num = len(x_test_noisy)
	plt.figure(figsize=(18, 4))    
	rdselect = rd.sample(range(num), n)
	print("n====",rdselect)
	for i in range(n):
		# display original images
		ax = plt.subplot(3, n, i + 1)
		plt.imshow(x_test[rdselect[i]].reshape(48, 48))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display noise images
		ax = plt.subplot(3, n, i + 1 + n)
		plt.imshow(decoded_imgs_cum[rdselect[i]].reshape(48, 48))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
        
		ax = plt.subplot(3, n, i + 1 + 2*n)
		plt.imshow(x_test_noisy[rdselect[i]].reshape(48, 48))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)        
	plt.show()
	sigma = sigma+0.35  #Ä©Î»Îª5±íÊ¾abs
	path = 'D:\\jupyter_notebook\\lena\\denoise48\\denoise_cum'+str(sigma)+'\\'
	immagedenoise(path,'denoise',num,decoded_imgs_cum)
	
	#path = 'D:\\jupyter_notebook\\lena\\denoise48\\denoise'+str(sigma)+'\\'
	#immagedenoise(path,'denoise',num,decoded_imgs)
	print(num)
