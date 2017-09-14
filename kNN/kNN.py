# -*-coding:utf-8-*-
import numpy as np
import operator


def classify0(inX, dataSet, labels, k):
    # 欧式距离公式
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # numpy的argsort()函数返回distances矩阵从小到大的索引值
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        # 依照索引值选定前k个对应标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # 通过字典设置标签出现频次
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 对字典排序，按照第二个域(即频次)，由大到小排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现频次最高的标签
    return sortedClassCount[0][0]
