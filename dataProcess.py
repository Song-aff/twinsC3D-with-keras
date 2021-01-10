import os
import cv2
import numpy as np
import random


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


def getMultiplesOfNotThree():  # 获取一个的非3的倍的数
    out = random.randint(0, listLen-1)
    while out % 3 == 0:
        out = random.randint(0, listLen-1)
    return out


def getBatchMultiplesOfNotThree(batch):  # 获取一组batch的非3的倍的数
    outlist = []
    for i in range(batch):
        outlist.append(getMultiplesOfNotThree())
    return(outlist)


def getMultiplesOfThree():  # 获取一个的3的倍的数
    return random.randint(1, int(listLen/3))*3


def getBatchMultiplesOfThree(batch):  # 获取一组batch的3的倍的数
    outlist = []
    for i in range(batch):
        outlist.append(getMultiplesOfThree())
    return(outlist)


def getLabels(x1_index, x2_index):
    if rollerList[x1_index]['lable'] == beltList[x2_index]['lable']:
        return 1.0
    else:
        return 0.0


def getFullrollerData(index1, train=True):
    while len(os.listdir(rollerList[index1]['dir'])) != 16:
        if train:
            index1 = getMultiplesOfNotThree()
        else:
            index1 = getMultiplesOfThree()
    return index1


def getFullBeltData(index1, train=True):
    while len(os.listdir(beltList[index1]['dir'])) != 16:
        if train:
            index1 = getMultiplesOfNotThree()
        else:
            index1 = getMultiplesOfThree()
    return index1


def process_batch(indexList1, indexList2, num, train=True):
    # 双读取
    batch1 = np.zeros((num, 16, 112, 112, 3), dtype='float32')
    batch2 = np.zeros((num, 16, 112, 112, 3), dtype='float32')
    labels = np.zeros((num, 1), dtype='float32')
    currentLable = True
    for i in range(num):
        temp1 = indexList1[i]
        temp2 = indexList2[i]
        for j in range(16):
            temp1 = getFullrollerData(temp1)
            temp2 = getFullBeltData(temp2)
            if currentLable:
                while getLabels(temp1, temp2) != 1.0:
                    if train:
                        temp1 = getFullrollerData(getMultiplesOfNotThree())
                        temp2 = getFullBeltData(getMultiplesOfNotThree())
                    else:
                        temp1 = getFullrollerData(getMultiplesOfThree())
                        temp2 = getFullBeltData(getMultiplesOfThree())
            else:
                while getLabels(temp1, temp2) != 0.0:
                    if train:
                        temp1 = getFullrollerData(getMultiplesOfNotThree())
                        temp2 = getFullBeltData(getMultiplesOfNotThree())
                    else:
                        temp1 = getFullrollerData(getMultiplesOfThree())
                        temp2 = getFullBeltData(getMultiplesOfThree())
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
        labels[i] = getLabels(temp1, temp2)
        # 保证数据集平衡
        currentLable = not currentLable

    return batch1, batch2, labels


def generator_train_batch(batch_size):
    while True:
        x1, x2, l = process_batch(
            getBatchMultiplesOfNotThree(batch_size), getBatchMultiplesOfNotThree(batch_size), batch_size)
        x1 = preprocess(x1)
        x2 = preprocess(x2)
        x1 = np.transpose(x1, (0, 2, 3, 1, 4))
        x2 = np.transpose(x2, (0, 2, 3, 1, 4))
        yield [x1, x2], l


def generator_validation_batch(batch_size):
    while True:
        x1, x2, l = process_batch(
            getBatchMultiplesOfThree(batch_size), getBatchMultiplesOfThree(batch_size), batch_size, train=False)
        x1 = preprocess(x1)
        x2 = preprocess(x2)
        x1 = np.transpose(x1, (0, 2, 3, 1, 4))
        x2 = np.transpose(x2, (0, 2, 3, 1, 4))
        yield [x1, x2], l
