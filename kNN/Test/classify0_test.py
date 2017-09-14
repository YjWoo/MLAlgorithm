#-*-coding:utf-8-*-
import numpy as np
from kNN import kNN
import matplotlib.pyplot as plt


def createDataSet():
    group=np.array([[1,2],[2,3],[4,2],[1,1],[3,1],[4.5,3]])
    labels=['A','A','B','A','B','B']
    return group,labels

group,labels=createDataSet()

x=[]
y=[]
#输入坐标
input=[3,2]
lists=group.tolist()

for i in lists:
    x.append(i[0])
    y.append(i[1])
plt.plot(x,y,'ro')
plt.plot(input[0],input[1],'bo')
#matplotlib添加注释
plt.annotate('Result',xy=(input[0],input[1]))
for i in range(6):
    plt.annotate(labels[i],xy=(x[i],y[i]),xytext=(x[i],y[i]-0.1))

result=kNN.classify0(input,group,labels,5)
str='kNN-->input:',input,'result:',result
plt.title(str)
plt.show()
