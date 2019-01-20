# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 13:32:19 2019

@author: xq
"""
#ÏÔ´æ²»¹»£¬Òª²ÉÓÃµü´úÆ÷This is the finall version
from keras import Input
import numpy as np
from keras import layers
from keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,\
    BatchNormalization,Flatten,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D, \
    UpSampling2D, Conv2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf  
import os 
import glob 
from keras import backend as K
import random  as rd
from PIL import Image
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
'''def cum_loss(y_true,y_pred):
    distance = y_pred - y_true
    mean = K.mean(distance)
    distance -= mean
    cum3_d = K.mean((distance * distance) * distance)
    cum4_d = K.mean(((distance * distance) * distance) * distance) - 3 * K.mean(distance * distance)* K.mean(distance * distance)
    #return K.mean(K.square(y_pred - y_true),axis=-1) +  20 * K.abs(cum3_d) + 5 * K.abs(cum4_d)
    #return K.mean(K.square(distance),axis=-1) + 10 * K.abs(cum3_d) + 5 * K.abs(cum4_d)
    return K.mean(K.square(distance),axis=-1) + 10* K.square(cum4_d) #40 *  K.square(cum3_d)  '''

def cum_loss1(y_true,y_pred):
    distance = y_pred - y_true
    mean = K.mean(distance)
    distance -= mean
    cum3_d = K.mean((distance * distance) * distance)
    cum4_d = K.mean(((distance * distance) * distance) * distance) - 3 * K.mean(distance * distance)* K.mean(distance * distance)
    #return K.mean(K.square(y_pred - y_true),axis=-1) +  20 * K.abs(cum3_d) + 5 * K.abs(cum4_d)
    #return K.mean(K.square(distance),axis=-1) + 10 * K.abs(cum3_d) + 5 * K.abs(cum4_d)
    return K.mean(K.square(distance),axis=-1)# + 40* K.square(cum4_d) #80 * K.square(cum3_d) +
 
def cum_coef(y_true, y_pred):
   
    distance = y_pred - y_true
    mean = K.mean(distance)
    distance -= mean
    Rx = K.mean(K.square(distance))
    global n 
    global cum
    cum3_d = K.mean((distance * distance) * distance)
    cum4_d = K.mean(((distance * distance) * distance) * distance) - 3 * K.square(Rx)
    #K.cast(cum3_d, dtype='float64')
    #print("MMMMM-========----------------------",K.is_keras_tensor(cum3_d),cum4_d)
    #cum_new = [K.eval(cum3_d),K.eval(cum4_d)] 
#    print("MMMMM-========----------------------",cum_new)	        
    if n:
        K.update(cum[0], cum3_d)
        K.update(cum[1], cum4_d)
        n = n - 1
       
    return K.mean(K.square(y_pred - y_true),axis=-1), cum3_d, cum4_d

def cum_loss(y_true, y_pred):
    normal_loss, cum3_d, cum4_d = cum_coef(y_true, y_pred)
    global cum 
    global coef
    cum_norm_old = K.sqrt(K.square(cum[0]))
    cum_norm_new = K.sqrt(K.square(cum4_d))
    
    if K.greater_equal(cum_norm_old/4.0, cum_norm_new) is not None:
        for n in range(1):
                if K.less(coef[n+1], coef[0] * cum[n]) is not None:
                    K.update(coef[n+1], K.zeros(shape=(1, 1)))
                else:
                    K.update(coef[n+1], coef[n+1] - coef[0] * cum[n])
        else:
            K.update(coef[0], 10 * coef[0])
    #print("MMMMM-========----------------------", K.dot(K.reshape(coef[1:3],(1,2)),K.reshape(cum,(2,1))))            
    loss = normal_loss - K.dot(K.reshape(coef[1:3],(1,2)),K.reshape(cum,(2,1))) + 0.5 * coef[0] * K.square(cum4_d)   
    #print("MMMMM-========----------------------",coef)  
    K.update(cum[0], cum3_d)
    K.update(cum[1], cum4_d)     
    return loss

def model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy,loss_fun='mean_squared_error'):
    # ¶¨Òåencoder
    input_img = Input(shape=(48, 48, 1))  # (?, 240, 240, 1)
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
    #net = Conv2D(32, (3, 3), activation='relu', padxing='same')(net) 
    net = Conv2D(16, (3, 3), activation='relu', padding='same')(net)  # (?, 240, 240, 16)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(net)  # (?, 48, 48, 1)
  
    # Ñ¡¶¨Ä£ÐÍµÄÊäÈë£¬decoded£¨¼´Êä³ö£©µÄ¸ñÊ½
    auto_encoder = Model(input_img, decoded)

    # ¶¨ÒåÓÅ»¯Ä¿±êºÍËðÊ§º¯Êý
    auto_encoder.compile(optimizer='adam', loss=loss_fun)
    #print("MMMMM-========----------------------")	        
    datagen = ImageDataGenerator(zca_whitening=True)

    validation_generator = datagen.flow(x_val_noisy, x_val, batch_size=32)

	
    # ÑµÁ·
    '''auto_encoder.fit(x_train_noisy, x_train,  # ÊäÈëÊä³ö
                     epochs=10,  # µü´ú´ÎÊý
                     batch_size=128,
                     shuffle=True,
                     validation_data=(x_test_noisy, x_test))  # ÑéÖ¤¼¯'''
    auto_encoder.fit_generator(datagen.flow(x_train_noisy, x_train, batch_size=32),
                    steps_per_epoch=len(x_train_noisy) / 32, epochs=50,                    
                    validation_data=validation_generator,
                    validation_steps=len(x_val_noisy) / 32)
					
    decoded_imgs = auto_encoder.predict(x_test_noisy)  # ²âÊÔ¼¯ºÏÊäÈë²é¿´Æ÷È¥ÔëÖ®ºóÊä³ö¡£
    return decoded_imgs


#¶ÁÈ¡ÔëÉùÍ¼Ïñ
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
    imgs = np.reshape(imgs,(num,48,48,1))
    imgs = imgs.astype('float32')/255
    #imgs -= np.mean(imgs)    
    return imgs

def immagedenoise(path,filename,n,decoded_imgs):
    isExists = os.path.exists(path)
    if not isExists:
         os.makedirs(path)
    for i in range(n):
        #print((decoded_imgs[i]))
        #out = Image.fromarray(np.uint8(decoded_imgs[i].reshape(48, 48)*255))
        #out.save(path+str(i+1)+'.png','PNG')   
        plt.imsave(path+str(i+1)+'.png',decoded_imgs[i].reshape(48, 48))	

		
if __name__ == '__main__':
	coef = K.variable(value=np.array([1000,3,3]), dtype='float32', name='init_coef')		
	sigma = 20			
	global cum    
	global n    
	n = 1		
	cum = K.variable(value=np.array([0,0]), dtype='float32', name='init_cum')						
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
	print("====",coef)    
    
	decoded_imgs_cum = model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy,loss_fun=cum_loss)
	print("#####",np.shape(decoded_imgs_cum))
	#decoded_imgs = model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy,loss_fun=cum_loss1)

	num = len(x_test_noisy)
	plt.figure(figsize=(18, 4))  
	n = 8    
	rdselect = rd.sample(range(num), n)
	print("n====",rdselect,coef)
	print("coef====",coef)        
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
	
	path = 'D:\\jupyter_notebook\\lena\\denoise48\\denoise'+str(sigma)+'\\'
	#immagedenoise(path,'denoise',num,decoded_imgs)
	print(num)
