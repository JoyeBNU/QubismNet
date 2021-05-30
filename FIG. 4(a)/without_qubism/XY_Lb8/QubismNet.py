#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import os.path
import csv
import cv2
import re
import time
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam
from keras.layers import Activation, Dropout, Flatten, Dense, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

t_start = time.time()

# Hyperparams
STATE_SIZE = 65536
EPOCHS = 300
BATCH_SIZE = 32
INPUT_SHAPE = (STATE_SIZE, 1)
TRAINING_LOGS_FILE = 'training_logs.csv'
MODEL_SUMMARY_FILE = 'model_summary.txt'
MODEL_NAME = 'qucnn_regression'

# 读取数据文件
pkl_file_1 = open('rdm1D_0.0_0.2994.pkl', 'rb')
data_train_1 = pickle.load(pkl_file_1)       # data: (500, 65536)
pkl_file_1.close()

pkl_file_2 = open('rdm1D_0.7_1.0000000000000224.pkl', 'rb')
data_train_2 = pickle.load(pkl_file_2)       # data: (500, 65536)
pkl_file_2.close()

pkl_file_3 = open('rdm1D_0.0_0.99.pkl', 'rb')
data_all = pickle.load(pkl_file_3)       # data: (100, 65536)
pkl_file_3.close()

pkl_label_1 = open('h_0.0_0.2994.pkl', 'rb')
h_1 = pickle.load(pkl_label_1)         # label: (500, 1)
pkl_label_1.close()

pkl_label_2 = open('h_0.7_1.0000000000000224.pkl', 'rb')
h_2 = pickle.load(pkl_label_2)         # label: (500, 1)
pkl_label_2.close()

pkl_label_3 = open('h_0.0_0.99.pkl', 'rb')
h_all = pickle.load(pkl_label_3)         # label: (100, 1)
pkl_label_3.close()

# 组织数据成为训练集和测试集
data_train = np.concatenate((data_train_1, data_train_2), axis=0)  # axis=0为纵向合并，axis=1为横向合并
data_test = data_all[30:70, :]

# 增加一个维度以符合CNN的输入要求
data_train, data_test = np.expand_dims(data_train, axis=2), np.expand_dims(data_test, axis=2)

# resize数据到[0, 1]之间
data_train = (data_train - data_train.min()) / (data_train.max() - data_train.min())
data_test = (data_test - data_test.min()) / (data_test.max() - data_test.min())

# 读取标签数据
h_train = np.concatenate((h_1, h_2), axis=0)
h_test = h_all[30:70]

# CNN Model
model = Sequential()

model.add(Conv1D(filters=32, kernel_size=9, padding='same', activation='relu', input_shape=INPUT_SHAPE, kernel_initializer='he_normal'))
model.add(Conv1D(filters=32, kernel_size=9, padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling1D(pool_size=4))

model.add(Conv1D(filters=64, kernel_size=9, padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(Conv1D(filters=64, kernel_size=9, padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling1D(pool_size=4))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))

model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001))

with open(MODEL_SUMMARY_FILE, 'w') as fh:
    model.summary(print_fn=lambda line: fh.write(line + '\n'))
plot_model(model, to_file='model.eps', show_shapes=True)  # 画出模型结构图，并保存成图片

# Training
tbCallBack = TensorBoard(log_dir='TB_logs', histogram_freq=0, write_graph=True, write_images=True)
# earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
save_best = ModelCheckpoint(MODEL_NAME+'_best.h5', verbose=1, save_best_only=True)
history = model.fit(data_train, h_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1,
                    shuffle=True, callbacks=[CSVLogger(TRAINING_LOGS_FILE, append=False, separator=';'), tbCallBack, save_best])
with open(MODEL_NAME + '.txt', 'w') as f:
    f.write(str(history.history))
model.save(MODEL_NAME + '.h5')
model.save_weights(MODEL_NAME + '_weights.h5')
del model
model = load_model(MODEL_NAME + '.h5')
best_model = load_model(MODEL_NAME+'_best.h5')

# 训练历史可视化：绘制训练 & 验证的损失值
fig11, axes11 = plt.subplots(figsize=(16, 10), dpi=600)
type11 = axes11.plot(history.history['loss'], color='blue', linewidth=3, linestyle='-.', label='Training')
type22 = axes11.plot(history.history['val_loss'], color='green', linewidth=3, linestyle='dashed', label='Validation')
plt.ylabel('Loss', fontdict={'size': 30})
plt.xlabel('Epoch', fontdict={'size': 30})
plt.yticks(fontproperties='Times New Roman', size=28)
plt.xticks(fontproperties='Times New Roman', size=28)
axes11.legend(prop={'size': 28})
plt.savefig(MODEL_NAME + '_loss.pdf')
plt.savefig(MODEL_NAME + '_loss.jpg')

# Testing
score_loss = model.evaluate(x=data_test, y=h_test, verbose=1)
score_loss_best = best_model.evaluate(x=data_test, y=h_test, verbose=1)
with open('test_result.txt', 'a') as fre:
    fre.write('test set mse is: ' + str(score_loss) + '\n')
    fre.write('test set mse base on best model is: ' + str(score_loss_best) + '\n')
    
# Testing test set for probabilities
h_pred = model.predict(data_test, verbose=1)
h_pred_best = best_model.predict(data_test, verbose=1)
test_h_preds = np.concatenate((h_test.reshape(-1, 1), h_pred, h_pred_best), axis=1)
np.savetxt('test_h_preds.txt', test_h_preds)

# Testing training set for probabilities
h_pred_tr = model.predict(data_train, verbose=1)
h_pred_tr_best = best_model.predict(data_train, verbose=1)
train_h_preds = np.concatenate((h_train.reshape(-1, 1), h_pred_tr, h_pred_tr_best), axis=1)
np.savetxt('train_h_preds.txt', train_h_preds)

# 计算程序运行总时间
t_end = time.time()
t_total = t_end-t_start
with open('test_result.txt', 'a') as fre:
    fre.write('Total time the program use is: ' + str(t_total) + '\n')
