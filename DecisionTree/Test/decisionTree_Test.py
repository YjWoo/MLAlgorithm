from DecisionTree import decisionTree as tree

dataset=[[1,1,'yes'],[1,2,'yes'],[1,0,'no'],[1,1,'yes']]

print(tree.DecisionTree.calcShannonEnt(dataset))

if __name__ == '__main__':
    data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'yes'], [0, 1, 'no'], [0, 1, 'no']]

    print(tree.DecisionTree.calcShannonEnt(data))
    print(tree.DecisionTree.chooseBestFeatureToSplit(data))