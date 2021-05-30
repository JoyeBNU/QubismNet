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
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

t_start = time.time()
TRAINING_LOGS_FILE = "training_logs.csv"
MODEL_SUMMARY_FILE = "model_summary.txt"
MODEL_NAME = 'qucnn_regression'
PIC_PATH = 'training_set'
TEST_PIC_PATH = 'generalizing_set'

# Hyperparams
IMAGE_SIZE = 512
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 500
BATCH_SIZE = 32
input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

# CNN Model
model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))

model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001))

with open(MODEL_SUMMARY_FILE, 'w') as fh:
    model.summary(print_fn=lambda line: fh.write(line + '\n'))
plot_model(model, to_file='model.eps', show_shapes=True)  # 画出模型结构图，并保存成图片


# Data
def process_file(picPath):
    X_list = []
    Y_list = []
    file_list = os.listdir(picPath)
    for i in range(len(file_list)):
        image = cv2.imread(os.path.join(picPath, file_list[i]))
        if file_list[i].endswith('.jpg'):
            img_ = np.array(image)
            img = img_ / 255.0
            X_list.append(img)
            B = re.search(r'\d+\.\d+', file_list[i])
            Y_list.append(eval(B.group()))
    X, Y = np.array(X_list), np.array(Y_list)
    index = list(range(len(Y)))
    random.shuffle(index)
    X = X[index]
    Y = Y[index]
    return X, Y, len(Y_list)


train_X, train_Y, num = process_file(PIC_PATH)

# Training
tbCallBack = TensorBoard(log_dir='TB_logs', histogram_freq=0, write_graph=True, write_images=True)
# earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
save_best = ModelCheckpoint(MODEL_NAME+'_best.h5', verbose=1, save_best_only=True)
history = model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1,
                    callbacks=[CSVLogger(TRAINING_LOGS_FILE, append=False, separator=';'), tbCallBack, save_best])
with open(MODEL_NAME + '.txt', 'w') as f:
    f.write(str(history.history))
model.save(MODEL_NAME + '.h5')
model.save_weights(MODEL_NAME + '_weights.h5')
del model
model = load_model(MODEL_NAME + '.h5')
best_model = load_model(MODEL_NAME+'_best.h5')

# 训练历史可视化：绘制训练 & 验证的损失值
fig3, axes3 = plt.subplots(figsize=(16, 10), dpi=600)
type7 = axes3.plot(history.history['loss'], color='blue', linewidth=3, linestyle='-.', label='Training')
type8 = axes3.plot(history.history['val_loss'], color='green', linewidth=3, linestyle='dashed', label='Validation')
plt.ylabel('Loss', fontdict={'size': 30})
plt.xlabel('Epoch', fontdict={'size': 30})
plt.yticks(fontproperties='Times New Roman', size=28)
plt.xticks(fontproperties='Times New Roman', size=28)
axes3.legend(prop={'size': 28})
plt.savefig(MODEL_NAME + '_loss.eps')

# Testing
test_X, test_Y, num_test = process_file(TEST_PIC_PATH)
score_loss = model.evaluate(x=test_X, y=test_Y, verbose=1)
score_loss_best = best_model.evaluate(x=test_X, y=test_Y, verbose=1)
with open('test_result.txt', 'a') as fre:
    fre.write('test set mse is: ' + str(score_loss) + '\n')
    fre.write('test set mse base on best model is: ' + str(score_loss_best) + '\n')
    fre.write('Total training number is: ' + str(num) + '\n')
    fre.write('Total test number is: ' + str(num_test) + '\n')

# Testing per image in test set
fileList = os.listdir(TEST_PIC_PATH)
num_valid = 0
for j in range(len(fileList)):
    image_test = cv2.imread(os.path.join(TEST_PIC_PATH, fileList[j]))
    if fileList[j].endswith('.jpg'):
        num_valid += 1
        img_test = np.array(image_test) / 255.0
        B_test = re.search(r'\d+\.\d+', fileList[j])
        predict_B = model.predict(np.array([img_test]))
        predict_B_best = best_model.predict(np.array([img_test]))
        with open('test_result.csv', 'a', newline='') as myFile:
            myWriter = csv.writer(myFile)
            myWriter.writerow([num_valid, fileList[j], eval(B_test.group()), predict_B[0][0], predict_B_best[0][0]])

# Testing per image in train set
fileList_ = os.listdir(PIC_PATH)
num_valid_ = 0
for k in range(len(fileList_)):
    image_train = cv2.imread(os.path.join(PIC_PATH, fileList_[k]))
    if fileList_[k].endswith('.jpg'):
        num_valid_ += 1
        img_train = np.array(image_train) / 255.0
        B_train = re.search(r'\d+\.\d+', fileList_[k])
        predict_B = model.predict(np.array([img_train]))
        predict_B_best = best_model.predict(np.array([img_train]))
        with open('train_result.csv', 'a', newline='') as myFile_:
            myWriter_ = csv.writer(myFile_)
            myWriter_.writerow([num_valid_, fileList_[k], eval(B_train.group()), predict_B[0][0], predict_B_best[0][0]])



# 计算程序运行总时间
t_end = time.time()
t_total = t_end-t_start
with open('test_result.txt', 'a') as fre:
    fre.write('Total time the program use is: ' + str(t_total) + '\n')
