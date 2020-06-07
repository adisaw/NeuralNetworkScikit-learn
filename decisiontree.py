import numpy as np 
from sklearn import tree

train=np.genfromtxt('train.csv',dtype=np.float64,delimiter=',')
test=np.genfromtxt('test.csv',dtype=np.float64,delimiter=',')

trainX=np.zeros(shape=(168,7))
trainX[:,:]=train[:,0:7]
trainY=np.zeros(168)
trainY=train[:,7]
testX=np.zeros(shape=(42,7))
testX[:,:]=test[:,0:7]
testY=np.zeros(42)
testY=test[:,7]

dt=tree.DecisionTreeClassifier()
dt.fit(trainX,trainY)
output=dt.predict(testX)

count=0
for i in range(test.shape[0]):
	if output[i]==testY[i]:
		count=count+1

print('Testing Accuracy ',count/test.shape[0])
