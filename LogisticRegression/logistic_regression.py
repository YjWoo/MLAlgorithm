# -*- coding: utf-8 -*-
# !/usr/bin/python

from numpy import *

# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# alpha:步长即学习速率，maxCycles:迭代次数，梯度上升法计算得到回归系数
def gradAscent(dataArray, labelArray, alpha, maxCycles):
    dataMat = mat(dataArray)  # size:m*n
    labelMat = mat(labelArray)  # size:m*1
    m, n = shape(dataMat)
    weigh = ones((n, 1))
    for i in range(maxCycles):
        h = sigmoid(dataMat * weigh)
        error = labelMat - h  # size:m*1
        weigh = weigh + alpha * dataMat.transpose() * error
    return weigh


# 根据回归系数对输入的样本进行预测
def classfy(weigh, dataArray, labelArray):
    dataMat = mat(dataArray)
    labelMat = mat(labelArray)
    h = sigmoid(dataMat * weigh)  # size:m*1
    m = len(h)
    error = 0.0
    for i in range(m):
        if int(h[i]) > 0.5:
            print(int(labelMat[i]), 'is classfied as: 1')
            if int(labelMat[i]) != 1:
                error += 1
                print('error')
        else:
            print(int(labelMat[i]), 'is classfied as: 0')
            if int(labelMat[i]) != 0:
                error += 1
                print('error')
    print('error rate is:', '%.4f' % (error / m))

