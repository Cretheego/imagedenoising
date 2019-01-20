# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:25:39 2018

@author: Administrator
"""

from PIL import Image
import sys
import os 
import shutil

def resample_image(img):
    width, height = image.size
    item_width = np.arange(int(width / 5))
	b = np.arange(width)
    for n in 5:
        for m in 5:
            out = img[,b[n::5]][b[m::5],:]
			out.save(targetfile)            

path = "K:/imagenet/originalPics" # "K:/imagenet/lfw/lfw"
targetpath = "K:/imagenet/originalPics/source_image"
#"K:/imagenet/source_image"
targetgray = "K:/imagenet/originalPics/source_gray_resize"
targetspit = "K:/imagenet/originalPics/source_gray_resize"
#"K:/imagenet/source_grayss" #s±íÊ¾ËõÐ¡µ½250X250
width = 240
height = 240
#±éÀúÎÄ¼þ¼Ð£¬È¡³öËùÓÐÍ¼Æ¬
for fpathe,dirs,fs in os.walk(path):
    #fs.sort(key=lambda x:int(x[n:-4]))
    #print(fpathe,dirs,fs)
    for i in range(len(fs)):
        #print(fs)
        srcfile = fpathe + '/' + fs[i]
        targetfile = targetgray+'/' + fs[i]
        shutil.copy(srcfile,targetpath) 
        img = Image.open(srcfile).convert('L')
        flag,img = resample_image(img)
        if flag:
            out = img.resize((width, height),Image.ANTIALIAS)
            out.save(targetfile)
    #print(fs)