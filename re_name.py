# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 20:01:52 2018

@author: Administrator
"""

from PIL import Image
import sys
import os 
import shutil
import string 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == '__main__':
    path = "D:\\jupyter_notebook\\lena\\clean"
    #遍历文件夹，取出所有图片
    if 1:
        for fpathe,dirs,fs in os.walk(path):
            #fs.sort(key=lambda x:int(x[n:-4]))
            print(fpathe,dirs,fs)
            j = 0
            for i in range(len(fs)):
                srcfile = fpathe + '/' + fs[i]
                newname = fs[i]
                newname = newname.split(".")
                if newname[-1]=="jpg":
                    newname[-1]="png"
                    newname = str.join(".",newname)  #这里要用str.join

                targetfile = fpathe + '/' + 'img' + str(j) + '.png'
                #shutil.copy(srcfile,targetpath + str(j) + fs[i]) 
                img = Image.open(srcfile)
                #out.save(targetfile)
                j+=1
                #print(j)
    
