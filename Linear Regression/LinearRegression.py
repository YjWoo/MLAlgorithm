# -*-coding:utf-8-*-
import numpy as np

def createDataset():
	x=np.array([1,2,3,6,4,6,7,8,9,10]).reshape(5,2)
	y=np.array([6,10,12,15,18]).reshape(5,1)
	return x,y

#线性回归，标准方程法Normal Equation求出参数
def NormalEquation(dataX,dataY):
	colNum=np.shape(dataX)[0]
	x0=np.ones((colNum,1))
	dataX=np.hstack((x0,dataX))
	#print(dataX)
	dataXT=dataX.T
	#print(dataXT)
	theta=dataXT.dot(dataX)
	#print(theta)
	#print(np.linalg.det(theta))
	theta=np.linalg.inv(theta)
	theta=theta.dot(dataXT)
	theta=theta.dot(dataY)
	return theta

if __name__=='__main__':
	x,y=createDataset()
	t=NormalEquation(x,y)
	print(t.flatten())
