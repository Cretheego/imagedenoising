# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:25:39 2018

@author: Administrator
"""

import numpy as np
from PIL import Image
import sys
import os 
import shutil
import random  as rd
import string 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#图像填充，重采样

#先将 input image 填充为正方形
def fill_image(image):
    width, height = image.size
    #选取长和宽中较大值作为新图片的
    new_image_length = width if width > height else height
    #生成新图片[白底]
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')   #注意这个函数！
    #将之前的图粘贴在新图上，居中
    if width > height:#原图宽大于高，则填充图片的竖直维度  #(x,y)二元组表示粘贴上图相对下图的起始位置,是个坐标点。
        new_image.paste(image, (0, int((new_image_length - height) / 2)))
    else:
        new_image.paste(image, (int((new_image_length - width) / 2),0))
    return new_image

def resample_image(img,filename):
    #width, height = image.size
    b = np.arange(width)
    ary_img = np.asarray(img)
    print(ary_img)
    i = 0
    for n in np.arange(5):
        for m in np.arange(5):
            ary_out = ary_img[:,b[n::5]][b[m::5],:]
            i += 1
            if n==0 and m==0:
                print((ary_out))
            out = Image.fromarray((ary_out))
            out.save(IMAGE_OUTPUT_PATH+str(i)+'.png','PNG')      
            
def resample_image_rd(img):
    width, height = img.size
    b = np.arange(width)
    ary_img = np.asarray(img)
    #print(b)
    i = 5
    n = rd.sample(range(i), 1)[0]
    m = rd.sample(range(i), 1)[0]
    print(n,m)
    ary_out = ary_img[:,b[n::i]][b[m::i],:]
    out = Image.fromarray((ary_out))
    return out

if __name__ == '__main__':
    path = "K:/imagenet/originalPics/1_gray" 
    targetpath = "D:\\jupyter_notebook\\lena\\clean48"
    IMAGE_OUTPUT_PATH = './split/'
    IMAGE_OUTPUT_PATH1 = './denoise/denoise_cum30.35'
    width = 48 #图片尺寸修改为48X48
    height = 48
    #遍历文件夹，取出所有图片
    if 0:
        for fpathe,dirs,fs in os.walk(path):
            #fs.sort(key=lambda x:int(x[n:-4]))
            #print(fpathe,dirs,fs)
            j = 0
            for i in range(len(fs)):
                srcfile = fpathe + '/' + fs[i]
                newname = fs[i]
                newname = newname.split(".")
                if newname[-1]=="jpg":
                    newname[-1]="png"
                    newname = str.join(".",newname)  #这里要用str.join

                targetfile = targetpath+'/' + 'img' + str(j) + '.png'
                #shutil.copy(srcfile,targetpath + str(j) + fs[i]) 
                img = Image.open(srcfile).convert('L')
                #img = fill_image(img)
                out = img.resize((width, height),Image.ANTIALIAS)
                out.save(targetfile)
                j+=1
                print(j)
    #用采样方法
    if 1:
        for fpathe,dirs,fs in os.walk(path):
            #fs.sort(key=lambda x:int(x[n:-4]))
            #print(fpathe,dirs,fs)
            j = 0
            for i in range(len(fs)):
                srcfile = fpathe + '/' + fs[i]
                newname = fs[i]
                newname = newname.split(".")
                if newname[-1]=="jpg":
                    newname[-1]="png"
                    newname = str.join(".",newname)  #这里要用str.join

                targetfile = targetpath+'/' + 'img' + str(j) + '.png'
                #shutil.copy(srcfile,targetpath + str(j) + fs[i]) 
                img = Image.open(srcfile).convert('L')
                #img = fill_image(img)
                out = resample_image_rd(img)
                out.save(targetfile)
                j+=1
                print(j)