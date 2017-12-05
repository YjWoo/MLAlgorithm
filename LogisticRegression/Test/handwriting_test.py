from os import listdir
from numpy import *
from LogisticRegression import logistic_regression as lr


# 读取所有文件，转化为矩阵返回
def loadData(direction):
    trainfileList = listdir(direction)
    m = len(trainfileList)
    dataArray = zeros((m, 1024))
    labelArray = zeros((m, 1))
    for i in range(m):
        returnArray = zeros((1, 1024))  # 每个txt文件形成的特征向量
        filename = trainfileList[i]
        fr = open('%s/%s' % (direction, filename))
        for j in range(32):
            lineStr = fr.readline()
            for k in range(32):
                returnArray[0, 32 * j + k] = int(lineStr[k])
        dataArray[i, :] = returnArray  # 存储特征向量
        filename0 = filename.split('.')[0]
        label = filename0.split('_')[0]
        labelArray[i] = int(label)  # 存储类别
    return dataArray, labelArray


def digitRecognition(trainDir, testDir, alpha=0.05, maxCycles=10):
    data, label = loadData(trainDir)
    weigh = lr.gradAscent(data, label, alpha, maxCycles)
    print(weigh.shape)
    dataArray, labelArray = loadData(testDir)
    lr.classfy(weigh,dataArray,labelArray)

digitRecognition('train', 'test')
