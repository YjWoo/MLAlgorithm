from DecisionTree import decisionTree as tree
from DecisionTree import treePlotter as treeplot

if __name__ == '__main__':
    data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels=['no surfacing','flippers']

    print(tree.calcShannonEnt(data))
    print(tree.chooseBestFeatureToSplit(data))

    myTree=tree.createTree(data,labels)
    print(myTree)

    treeplot.createPlot(myTree)

