import cv2
import numpy as np
import os
import shutil
import time
import copy
import random
import tensorflow as tf
keras = tf.keras

trainFile = r".\train.list"
lable_list = []
with open(lable, 'r') as f:
    while True:
        temp = f.readline()
        if not temp:
            break
        lable_list.append(temp[:-1].split('-')[-1])
# print(lable_list)
# print(len(lable_list))


def my_generator(strat,
                 end,
                 img_dir=ImgDir,
                 batch=20,
                 img_size=(112, 112),
                 class_num=1):
    while True:
        out_imgs_1 = np.zeros([batch, img_size[0], img_size[1], 3], np.float)
        out_imgs_2 = np.zeros([batch, img_size[0], img_size[1], 3], np.float)
        out_imgs_3 = np.zeros([batch, img_size[0], img_size[1], 3], np.float)
        out_lable = np.zeros([batch, class_num], np.float)
        rand_num = []
        for i in range(batch):
            rand_num.append(random.randint(strat, end))
        # 保证标签满足要求
        for i in range(len(rand_num)):
            if lable_list[rand_num[i]] == lable_list[rand_num[i] + 1]:
                continue
            if lable_list[rand_num[i]] == lable_list[rand_num[i] - 1]:
                rand_num[i] = rand_num[i] - 1
                continue

            while True:
                rand_num[i] = random.randint(strat, end)
                if lable_list[rand_num[i]] == lable_list[rand_num[i] + 1]:
                    break
                if lable_list[rand_num[i]] == lable_list[rand_num[i] - 1]:
                    rand_num[i] = rand_num[i] - 1
                    break

        for i in range(len(rand_num)):
            out_lable[i] = keras.utils.to_categorical(
                (int(lable_list[rand_num[i]]) / 10.0), class_num)
            #out_lable[i] = rand_num[i]
            out_imgs_1[i] = cv2.resize(
                cv2.imread(ImgDir + str(rand_num[i]) + '.jpg'),
                (img_size[1], img_size[0])) / 255.0
            out_imgs_2[i] = cv2.resize(
                cv2.imread(ImgDir + str(rand_num[i] + 1) + '.jpg'),
                (img_size[1], img_size[0])) / 255.0
            out_imgs_3[i] = np.abs(out_imgs_1[i] - out_imgs_2[i])

            rota_flag = random.randint(0, 10)  # 随机旋转功能
            if rota_flag > 5:
                (h, w) = img_size
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, random.randint(-20, 20),
                                            1.0)
                out_imgs_1[i] = cv2.warpAffine(out_imgs_1[i], M, (w, h))
                out_imgs_2[i] = cv2.warpAffine(out_imgs_2[i], M, (w, h))
                out_imgs_3[i] = cv2.warpAffine(out_imgs_3[i], M, (w, h))
            # cv2.imshow("1",out_imgs[i][0])
            # cv2.imshow("2",out_imgs[i][1])
            # cv2.waitKey()
            # print(out)
        # print(rand_num[0])
        yield [out_imgs_1, out_imgs_2, out_imgs_3], out_lable

# return my_generator
