# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:30:22 2020

@author: Leo
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# 主函数
#import os
import cv2
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers.core import Activation,Dense
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, PReLU, concatenate,Input, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

def img_dilating(img, kernel_size):
    '''图像膨胀'''
    if kernel_size !=0:
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        img = cv2.dilate(img, kernel)
    return img

def array_img_reshape(array, target_shape):
    '''array图像缩放，要求三通道'''
    array = image.img_to_array(image.array_to_img(array).resize(target_shape))
    return array

def batch_generater(data_path, data_list, batch_size, shape, n_label):
    offset = 0
    while True:
        load_list = shuffle(data_list)
        X = np.zeros((batch_size, *shape))
        Y = np.zeros((batch_size, *shape[0:2], n_label))
        Z = np.zeros((batch_size, n_label))
        for i in range(batch_size):
            img_x_path = data_path[0] + '/' + load_list[i + offset]
            img_y_path = data_path[1] + '/' + load_list[i + offset]
            img_class  = int(load_list[i + offset][0])
            # load_img, img_to_array
            img_x = image.load_img(img_x_path, target_size = shape[0:2])
            img_y = image.load_img(img_y_path, target_size = shape[0:2])
            img_x = image.img_to_array(img_x)
            img_y = image.img_to_array(img_y)
            # 对label裂纹做膨胀和缩放
            #img_y = img_dilating(img_y, kernel_size=4) # kernel_size=0 时不膨胀
            #img_y = array_img_reshape(img_y, (62,62))
            # 取单通道, Normalization
            img_x = img_x[...,2:]
            img_y = img_y[...,2:]
            img_x = 1/255.0 * img_x
            img_y = 1/255.0 * img_y
            # one-hot 编码
            img_class = to_categorical(img_class,n_label)
            img_y = to_categorical(img_y, n_label)
            
            X[i,...] = img_x
            Y[i,...] = img_y
            Z[i,...] = img_class
            
            if i+offset >= len(load_list)-1:
                load_list = shuffle(data_list)
                offset = 0
        yield (X, {'seg_output': Y, 'class_output': Z})
        offset += batch_size

def conv(x, n_kernel, s_kernel, name_id, training=True, bn=True):
    x = Conv2D(n_kernel, s_kernel, strides=(1,1), padding='same', 
               kernel_initializer='he_normal', activation='relu', 
               name=str(name_id), trainable=training)(x)
    if bn:
        x = BatchNormalization()(x)
    return x

def build_model_up(input_shape, n_label):
    input_img = Input(shape=input_shape)
    # segmentation network
    seg_conv1 = conv(input_img, 32, 5, 'seg_conv1_01')
    seg_conv1 = conv(seg_conv1, 32, 5, 'seg_conv1_02')   # 500
    seg_maxp1 = MaxPooling2D(pool_size=(2,2))(seg_conv1) # 250
    seg_conv2 = conv(seg_maxp1, 64, 5, 'seg_conv2_01')   # 250
    seg_conv2 = conv(seg_conv2, 64, 5, 'seg_conv2_02')
    seg_conv2 = conv(seg_conv2, 64, 5, 'seg_conv2_03')   # 250
    seg_maxp2 = MaxPooling2D(pool_size=(2,2))(seg_conv2) # 125
    seg_conv3 = conv(seg_maxp2, 64, 5, 'seg_conv3_01')   # 125
    seg_conv3 = conv(seg_conv3, 64, 5, 'seg_conv3_02')
    seg_conv3 = conv(seg_conv3, 64, 5, 'seg_conv3_03')
    seg_conv3 = conv(seg_conv3, 64, 5, 'seg_conv3_04')   # 125
    seg_maxp3 = MaxPooling2D(pool_size=(2,2))(seg_conv3) # 62
    seg_conv4 = conv(seg_maxp3, 1024, 5, 'seg_conv4_01') # 62
    seg_mask  = Conv2D(n_label, 1, strides=(1,1), padding='same', 
                       kernel_initializer='he_normal', activation='softmax', 
                       name='seg_mask')(seg_conv4)
    
    seg_upsp5 = UpSampling2D()(seg_conv4) # 124
    seg_upsp5 = ZeroPadding2D(padding=((0, 1), (0, 1)))(seg_upsp5) # 125
    seg_upsp5 = concatenate([seg_upsp5,seg_conv3],axis=3)
    seg_conv5 = conv(seg_upsp5, 64, 5, 'seg_conv5_01')
    #seg_conv5 = conv(seg_upsp5, 64, 5, 'seg_conv5_02')
    #seg_conv5 = conv(seg_upsp5, 64, 5, 'seg_conv5_03')
    #seg_conv5 = conv(seg_upsp5, 64, 5, 'seg_conv5_04')   # 125
    
    seg_upsp6 = UpSampling2D()(seg_conv5) # 256
    seg_upsp6 = concatenate([seg_upsp6,seg_conv2],axis=3)
    seg_conv6 = conv(seg_upsp6, 64, 5, 'seg_conv6_01')
    #seg_conv6 = conv(seg_upsp6, 64, 5, 'seg_conv6_02')
    #seg_conv6 = conv(seg_upsp6, 64, 5, 'seg_conv6_03')   # 250
    
    seg_upsp7 = UpSampling2D()(seg_conv6) # 512
    seg_upsp7 = concatenate([seg_upsp7,seg_conv1],axis=3)
    seg_conv7 = conv(seg_upsp7, 32, 5, 'seg_conv7_01')
    #seg_conv7 = conv(seg_upsp7, 32, 5, 'seg_conv7_02')   # 500
    
    seg_output  = Conv2D(n_label, 1, strides=(1,1), padding='same', 
                         kernel_initializer='he_normal', activation='softmax', 
                         name='seg_output')(seg_conv7)
    # decision network
    dec_conv1 = concatenate([seg_conv4,seg_mask],axis=3)
    dec_conv1 = MaxPooling2D(pool_size=(2,2))(dec_conv1)
    dec_conv2 = conv(dec_conv1, 16, 5, 'dec_conv2_01')
    dec_conv2 = MaxPooling2D(pool_size=(2,2))(dec_conv2)
    dec_conv3 = conv(dec_conv2, 32, 5, 'dec_conv3_01')
    dec_conv3 = MaxPooling2D(pool_size=(2,2))(dec_conv3)
    dec_conv4 = conv(dec_conv3, 64, 5, 'dec_conv4_01')
    # classification
    class1 = GlobalMaxPooling2D()(dec_conv4)
    class2 = GlobalAveragePooling2D()(dec_conv4)
    class3 = GlobalAveragePooling2D()(seg_mask)
    class4 = GlobalMaxPooling2D()(seg_mask)
    
    classes = concatenate([class1,class2,class3,class4],axis=1)
    class_output = Dense(n_label, activation='softmax', name='class_output')(classes)

    model = Model(inputs=input_img, outputs=[seg_output,class_output])
    return model

def show_result(img_id, input_shape):
    src_path = './dataset/val_set/src/1_'+str(img_id)+'.png'
    lab_path = './dataset/val_set/label/1_'+str(img_id)+'.png'
    img1 = image.load_img(src_path, target_size = input_shape[0:2])  # src原图
    img2 = image.load_img(lab_path, target_size = input_shape[0:2])  # label原图
    img = image.img_to_array(img1)[...,2:]
    img = img/255
    img = np.expand_dims(img,axis=0)
    result = model.predict(img)
    fig  = result[0][0,...,1]
    fig1 = result[0][0,...]
    fig1 = np.reshape(fig1,(1,500*500,2))
    fig1 = np.argmax(fig1,axis=2).astype(np.int8)
    fig1 = np.reshape(fig1,(500,500))
    print('Result:',np.rint(result[1][0]))
    plt.figure(figsize=(8,8))
    plt.subplot(221)
    plt.imshow(img1)
    plt.subplot(222)
    plt.imshow(img2)
    plt.subplot(223)
    plt.imshow(fig)
    plt.subplot(224)
    plt.imshow(fig1)
    plt.show()
    return None

if __name__ == '__main__':
    input_shape = (500,500,1)
    n_label = 2
    batch_size = 2
    epoch = 5
    
    train_set_path  = ['./dataset/train_set/src','./dataset/train_set/label']
    valid_set_path  = ['./dataset/val_set/src'  ,'./dataset/val_set/label'  ]
    model_savepath  = './model/'
    model_savename  = 'best_model_up_500'
    train_name_list = os.listdir(train_set_path[0])
    valid_name_list = os.listdir(valid_set_path[0])
    
    model = build_model_up(input_shape, n_label)
    adam  = Adam(lr=0.001, epsilon=1e-08, decay=1e-5, amsgrad=True)
    model.compile(optimizer=adam,
                  metrics= {'seg_output': 'mse', 'class_output': 'acc'},
                  loss   = {'seg_output': 'mse', 'class_output': 'binary_crossentropy'},
                  loss_weights= {'seg_output': 1.0, 'class_output': 0.5})
    '''
    训练步骤：
    1. activation='softmax', loss = {'seg_output':'mse', 'class_output':'binary_crossentropy'}, epoch = 10
    2. activation='sigmoid', loss = {'seg_output':'binary_crossentropy', 'class_output':'binary_crossentropy'}, epoch = 10
    '''
    model.summary()
    
    # 保存模型网络结构
    with open(model_savepath+model_savename+'.json', 'w') as f:
        f.write(model.to_json())
    
    # 以加载权重的方式加载之前训练的结果
    if os.listdir(model_savepath):
        try: 
            model.load_weights(model_savepath + model_savename + '_weights.h5')
            print('Model weights loaded!')
        except:
            print('Model weights load filed!')
    
    save_best_1 = ModelCheckpoint(model_savepath + model_savename + '_weights.h5', monitor='val_seg_output_loss', 
                                  verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    save_best_2 = ModelCheckpoint(model_savepath + model_savename + '_weights.h5', monitor='val_class_output_acc', 
                                  verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    h = model.fit_generator(generator=batch_generater(train_set_path,train_name_list,batch_size,input_shape,n_label), 
                            steps_per_epoch = len(train_name_list)//batch_size, 
                            epochs=epoch, verbose=1, callbacks=[save_best_1, save_best_2, early_stop], 
                            validation_steps = len(valid_name_list)//8,
                            validation_data=batch_generater(valid_set_path,valid_name_list,8,input_shape,n_label))
    plt.figure(1) # 图 1 画 seg_output_mse
    plt.plot(h.history['seg_output_mean_squared_error'])
    plt.plot(h.history['val_seg_output_mean_squared_error'])
    plt.title('seg_output_mse')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('train_val_loss.jpg',dpi=600)
    plt.figure(2) # 图 2 画 class_output_acc
    plt.plot(h.history['class_output_acc'])
    plt.plot(h.history['val_class_output_acc'])
    plt.title('class_output_acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('train_val_acc.jpg',dpi=600)
    
    show_result(np.random.randint(0,500),input_shape)
    