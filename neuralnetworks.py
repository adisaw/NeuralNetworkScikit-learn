import numpy as np 
import warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier

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

mlp=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',solver='sgd',batch_size=32,learning_rate_init=0.01,max_iter=200)
mlp.fit(trainX,trainY)
output=mlp.predict(testX)

count=0
for i in range(test.shape[0]):
	if output[i]==testY[i]:
		count=count+1

print('Testing Accuracy :',count/test.shape[0])