import matplotlib.pyplot as plt
from numpy import *
from LogisticRegression import logistic_regression as lr


# 画出散点图分布
def plotBestFit(title,weigh, dataArray, labelArray):
    n = shape(dataArray)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if (int(labelArray[i]) >= 1):
            xcord1.append(dataArray[i][1]);
            ycord1.append(dataArray[i][2])
        else:
            xcord2.append(dataArray[i][1]);
            ycord2.append(dataArray[i][2])
    fig = plt.figure(title)
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='P')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(0.0, 10.0, 0.1)
    y = -x * float(weigh[2] / weigh[1])-float(weigh[0]/weigh[1])
    ax.plot(x, y)
    plt.show()


def loadDataSet(filename):
    fr = open(filename)
    lines = fr.readlines()
    m = len(lines)
    dataMat = zeros((m, 3));
    labelMat = zeros((m, 1))

    for i in range(m):
        lineArr = lines[i].strip().split()
        dataMat[i, 0] = 1
        dataMat[i, 1] = float(lineArr[0])
        dataMat[i, 2] = float(lineArr[1])
        labelMat[i, 0] = int(lineArr[2])
    return dataMat, labelMat


dataMat, labelMat = loadDataSet('matrix2d.txt')
dataTest, labelTest = loadDataSet('matrix2d_test.txt')
weigh = lr.gradAscent(dataMat, labelMat, 0.2, 10)
lr.classfy(weigh, dataTest, labelTest)
plotBestFit("训练数据",weigh, dataMat, labelMat)
plotBestFit("测试数据",weigh, dataTest, labelTest)
