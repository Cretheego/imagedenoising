# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# 本程序主要比较均值作为约束项与没有该约束项下的比较，初步结果表明在噪声较大的时候，
# 均值的权重较大，效果更好 80-20；125

from keras import Input
import numpy as np
from keras import layers
from keras.layers import Activation,\
    BatchNormalization,MaxPooling2D, \
    UpSampling2D, Conv2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf 
import time 
import os 
import glob 
from keras import backend as K
import random  as rd
from PIL import Image
from keras.initializers import glorot_uniform
from PIL import Image 
K.set_image_data_format("channels_last")
K.set_learning_phase(1)
import scipy.io as io

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
    history = auto_encoder.fit_generator(datagen.flow(x_train_noisy, x_train, batch_size=32),
                    steps_per_epoch=len(x_train_noisy) / 32, epochs=20,                    
                    validation_data=validation_generator,
                    validation_steps=len(x_val_noisy) / 32)
    start = time.clock()					
    decoded_imgs = auto_encoder.predict(x_test_noisy)  # ²âÊÔ¼¯ºÏÊäÈë²é¿´Æ÷È¥ÔëÖ®ºóÊä³ö¡£
    elapsed = (time.clock() - start)
    print("Time used:",elapsed/1000,history,history.history)
    #np.save('Loss_cum_1.npy', history.history['loss'])
    #np.save('Val_Loss_cum_1.npy', history.history['val_loss'])	
    #mat_path = 'your_mat_save_path'
    io.savemat('./Loss_cum_20.mat', {'Loss_cum_20': history.history['loss']})	
    io.savemat('./Val_Loss_cum_20.mat', {'Val_Loss_cum_20': history.history['val_loss']})	
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()	
    return decoded_imgs

#读取噪声图像
def trainset(Datapath,n):
    num = len(glob.glob(Datapath+'*.png'))
    print(num)
    imgs = np.zeros((num, 28, 28))
    i = 0
    for imageFile in glob.glob(Datapath+'*.png'):
        #print(imageFile)
        img = np.array(Image.open(imageFile))
        #print(np.shape(img))
        imgs[i] = img
        i += 1
    print(i)
    imgs = np.reshape(imgs,(num,28,28,1))
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
        plt.imsave(path+str(i+1)+'.png',decoded_imgs[i].reshape(28, 28))	

if __name__ == '__main__':
	coef = K.variable(value=np.array([1000,3,3]), dtype='float32', name='init_coef')		
	sigma = 50			
	global cum    
	global n    
	n = 1		
	cum = K.variable(value=np.array([0,0]), dtype='float32', name='init_cum')						
	dataset_clean = 'D:/jupyter_notebook/minist/clean/val/' 
	dataset_noisy = 'D:/jupyter_notebook/minist/noisy/gauss/val/'+'noisy_val'+str(sigma+0.3)+'/'
	x_val = trainset(dataset_clean,3)
	x_val_noisy = trainset(dataset_noisy,3)
	
	dataset_clean = 'D:/jupyter_notebook/minist/clean/train/' 
	dataset_noisy = 'D:/jupyter_notebook/minist/noisy/gauss/train/'+'noisy_train'+str(sigma+0.3)+'/'
	x_train = trainset(dataset_clean,3)
	x_train_noisy = trainset(dataset_noisy,3)
    
	dataset_clean = 'D:/jupyter_notebook/minist/clean/test/'     
	dataset_noisy = 'D:/jupyter_notebook/minist/noisy/gauss/test/'+'noisy_test'+str(sigma+0.3)+'/'
	x_test = trainset(dataset_clean,3)
	x_test_noisy = trainset(dataset_noisy,3)
	
	print("====",np.shape(x_train))
	print("====",np.shape(x_train_noisy))
	print("====",np.shape(x_test_noisy))
	print("====",coef)    
    
	decoded_imgs_cum = model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy,loss_fun=cum_loss)
	#print("#####",np.shape(decoded_imgs_cum))
	#decoded_imgs = model_train(x_train_noisy,x_train,x_val_noisy,x_val,x_test_noisy)

	num = len(x_test_noisy)
	plt.figure(figsize=(18, 4))  
	n = 6    
	rdselect = rd.sample(range(num), n)
	print("n====",rdselect,coef)
	print("coef====",K.eval(coef),K.eval(cum))        
	for i in range(n):
		# display original images
		ax = plt.subplot(3, n, i + 1)
		plt.imshow(x_test[rdselect[i]].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display noise images
		ax = plt.subplot(3, n, i + 1 + n)
		plt.imshow(decoded_imgs_cum[rdselect[i]].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
        
		ax = plt.subplot(3, n, i + 1 + 2*n)
		plt.imshow(x_test_noisy[rdselect[i]].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)        
	plt.show()
	
	path = 'D:\\jupyter_notebook\\minist\\denoise\\denoise_cum'+str(sigma+0.35)+'\\'
	immagedenoise(path,'denoise',num,decoded_imgs_cum)
	
	#path = 'D:\\jupyter_notebook\\minist\\denoise\\denoise_normal'+str(sigma)+'\\'
	#immagedenoise(path,'denoise',num,decoded_imgs)
	print(num)		
