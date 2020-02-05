# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:06:21 2020

@author: Leo
"""

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image



def load_model_from_json(model_path, model_name):
    with open(model_path+model_name+'.json', 'r') as file:
        model_json = file.read()
    model = model_from_json(model_json)
    try:
        model.load_weights(model_path+model_name+'_weights.h5')
        return model
    except:
        raise ValueError('Model weights mismatch!')

def array_img_reshape(array, target_shape):
    import PIL
    # PIL.Image.BILINEAR
    array = image.img_to_array(image.array_to_img(array).resize(target_shape))#, resample=PIL.Image.HAMMING))
    # "nearest", "bilinear", and "bicubic""lanczos""box" and"hamming"
    return array

def img_dilating(img, kernel_size):
    '''图像腐蚀'''
    if kernel_size !=0:
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        img = cv2.erode(img, kernel)
    return img

def preprocess(img, input_size):
    h,w = np.shape(img)
    img = img.resize(input_size)#, resample=PIL.Image.BILINEAR)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img, h, w

def process_output(result, shape, savepath, show_result=True):
    img_possibility = result[1][0,1] # 含有缺陷的 possibility 
    img_class = int(np.rint(img_possibility))
    print("Possibility of '{}' is: {:.4f}, takes {:.4f}s.".format(test_list[i], img_possibility, tok-tik))
    # 处理seg_output
    img_seg = result[0][0,...,1:]
    img_seg = np.repeat(img_seg, 3, axis=2)
    img_seg = array_img_reshape(img_seg, shape)
    #img_seg = img_dilating(img_seg, 3)
    img_seg = image.array_to_img(img_seg)
    # 保存输出图片
    savepath = savepath[0]+'class_{}_'.format(img_class)+savepath[1]
    #img_seg.save(savepath)
    if show_result:
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.imshow(omg)
        plt.subplot(122)
        plt.imshow(img_seg, cmap='gray')
        plt.show()
    return None

if __name__ == '__main__':
    # 设置模型的输入形状
    input_size  = (500, 500)
    output_size = (500, 500)
    n_label = 2
    
    # 加载模型 / 权重
    model_path = './model/'
    model_name = 'best_model_up_500'
    model = load_model_from_json(model_path, model_name)
    model.summary()
    
    test_path = './testdata'
    test_list = os.listdir(test_path)
    save_path = './result/'

    for i in range(len(test_list)):
        filepath = test_path + '/' + test_list[i]
        omg = image.load_img(filepath, grayscale=True)
        img, h, w = preprocess(omg, input_size)
        # 计时开始预测
        tik = time.time()
        result = model.predict(img, verbose=0)
        tok = time.time()
        # 进一步处理输出
        process_output(result, (w,h), [save_path, test_list[i]],show_result=True)

    
    
    
    
    
    
    
    