import os
import cv2
import numpy as np
import random
import tensorflow as tf
np_utils = tf.keras.utils

rollerList = []
# 属性包括 dir路径，lable 标签
beltList = []
listLen = 0
# 读取list文件
with open('roller_train.list', 'r', encoding='utf-8') as fp:
    temp = fp.readline()
    # print(os.listdir(temp.split(' ')[0]))
    # print(temp.split(' ')[1][:-1])
    rollerList.append(
        {'dir': temp.split(' ')[0], 'lable': temp.split(' ')[1][:-1]})
    while temp:
        temp = fp.readline()
        if len(temp) > 0:
            rollerList.append(
                {'dir': temp.split(' ')[0], 'lable': temp.split(' ')[1][:-1]})
    fp.close()
with open('belt_train.list', 'r', encoding='utf-8') as fp:
    temp = fp.readline()
    # print(os.listdir(temp.split(' ')[0]))
    # print(temp.split(' ')[1][:-1])
    beltList.append(
        {'dir': temp.split(' ')[0], 'lable': temp.split(' ')[1][:-1]})
    while temp:
        temp = fp.readline()
        if len(temp) > 0:
            beltList.append(
                {'dir': temp.split(' ')[0], 'lable': temp.split(' ')[1][:-1]})
    fp.close()
listLen = len(beltList)

# 包含中文路径使用 imdecode 代替 imread
# image = cv2.imdecode(np.fromfile(
#     rollerList[1]['dir']+'/' +
#     os.listdir(rollerList[1]['dir'])[0], dtype=np.uint8), -1)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (112, 112))
# print(image.shape)
# cv2.imshow('', image)
# cv2.waitKey()


def preprocess(inputs):
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    # inputs /=255.
    # inputs -= 0.5
    # inputs *=2.
    return inputs


def process_batch(start1, start2, num, train=True):
    # 双读取
    batch1 = np.zeros((num, 16, 112, 112, 3), dtype='float32')
    batch2 = np.zeros((num, 16, 112, 112, 3), dtype='float32')
    labels = np.zeros((num, 1), dtype='float32')
    for i in range(num):
        if train:
            temp1 = start1+i
            temp2 = start2+i
            for j in range(16):
                while len(os.listdir(rollerList[temp1]['dir'])) != 16:
                    temp1 = random.randint(0, listLen-1)
                while len(os.listdir(beltList[temp2]['dir'])) != 16:
                    temp2 = random.randint(0, listLen-1)
                image1 = cv2.imdecode(np.fromfile(
                    rollerList[temp1]['dir']+'/' +
                    os.listdir(rollerList[temp1]['dir'])[j], dtype=np.uint8), -1)
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                image1 = cv2.resize(image1, (112, 112))
                batch1[i][j][:][:][:] = image1

                image2 = cv2.imdecode(np.fromfile(
                    beltList[temp2]['dir']+'/' +
                    os.listdir(beltList[temp2]['dir'])[j], dtype=np.uint8), -1)
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                image2 = cv2.resize(image2, (112, 112))
                batch2[i][j][:][:][:] = image2
            if rollerList[temp1]['lable'] == beltList[temp2]['lable']:
                labels[i] = 1.0
            else:
                labels[i] = 0.0
        else:
            pass
    return batch1, batch2, labels


def generator_train_batch(batch_size):
    while True:
        x1, x2, l = process_batch(random.randint(
            0, listLen-1-batch_size), random.randint(0, listLen-1-batch_size), batch_size)
        x1 = preprocess(x1)
        x2 = preprocess(x2)
        x1 = np.transpose(x1, (0, 2, 3, 1, 4))
        x2 = np.transpose(x2, (0, 2, 3, 1, 4))
        yield [x1, x2], l


def generator_validation_batch(batch_size):
    while True:
        x1, x2, l = process_batch(random.randint(
            0, listLen-1-batch_size), random.randint(0, listLen-1-batch_size), batch_size)
        x1 = preprocess(x1)
        x2 = preprocess(x2)
        x1 = np.transpose(x1, (0, 2, 3, 1, 4))
        x2 = np.transpose(x2, (0, 2, 3, 1, 4))
        yield [x1, x2], l


