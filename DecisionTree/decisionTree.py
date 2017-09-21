# -*-coding:utf-8-*-
from math import log


# 计算给定数据集合的香农熵
def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currenLabel = featVec[-1]
        if currenLabel not in labelCounts.keys():
            labelCounts[currenLabel] = 0
        labelCounts[currenLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVect in dataSet:
        if featVect[axis] == value:
            reducedFeatVec = featVect[:axis]
            reducedFeatVec.extend(featVect[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分
def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):
        feaList = [example[i] for example in dataset]
        print(feaList)
        uniqueVals = set(feaList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset, i, value)
            prob = len(subDataSet) / float(len(dataset))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 返回出现次数最多的分类名称
def majorityCnt(classlist):
    classCount = {}
    for vote in classlist:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items, key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建树
def createTree(dataSet, lables):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLable = lables[bestFeat]
    myTree = {bestFeatLable: {}}
    del (lables[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLables = lables[:]
        myTree[bestFeatLable][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLables)
    return myTree


# 决策树分类函数
def classify(inputTree, featLabels, testVec):
    # 得到树中的第一个特征
    firstStr = inputTree.keys()[0]
    # 得到第一个对应的值
    secondDict = inputTree[firstStr]
    # 得到树中第一个特征对应的索引
    featIndex = featLabels.index(firstStr)
    # 遍历树
    for key in secondDict.keys():
        # 如果在secondDict[key]中找到testVec[featIndex]
        if testVec[featIndex] == key:
            # 判断secondDict[key]是否为字典
            if type(secondDict[key]).__name__ == 'dict':
                # 若为字典，递归的寻找testVec
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                # 若secondDict[key]为标签值，则将secondDict[key]赋给classLabel
                classLabel = secondDict[key]
                # 返回类标签
    return classLabel


# 决策树的序列化
def storeTree(inputTree, filename):
    # 导入pyton模块
    import pickle
    # 以写的方式打开文件
    fw = open(filename, 'w')
    # 决策树序列化
    pickle.dump(inputTree, fw)


# 读取序列化的树
def grabTree(filename):
    import pickle
    fr = open(filename)
    # 返回读到的树
    return pickle.load(fr)
