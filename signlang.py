# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("data/sign_mnist_train.csv")
test = pd.read_csv("data/sign_mnist_test.csv")
trainx = train.iloc[:,1:].values
trainy = train.iloc[:,0].values
testx = test.iloc[:,1:].values
testy = test.iloc[:,0].values

sc = StandardScaler()
trainx = sc.fit_transform(trainx)
testx = sc.fit_transform(testx)

oh = OneHotEncoder()
trainy = trainy.reshape(-1,1)
trainy = oh.fit_transform(trainy).toarray()
testy = testy.reshape(-1,1)
testy = oh.fit_transform(testy).toarray()

trainx = np.reshape(trainx, (27455, 28, 28, 1))
testx = np.reshape(testx, (7172, 28, 28, 1))


def create_placeholders(nh0, nw0, nc0, ny):
	x = tf.placeholder("float", shape = (None, nh0, nw0, nc0))
	y = tf.placeholder("float", shape = (None, ny))
	return x,y

def init_weights():
	w1 = tf.get_variable('w1', [3,3,1,16], initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = 2))
	
	params = {"w1":w1}
	return params


def forward_prop(x, params):
	w1 = params["w1"]
	

	z1 = tf.nn.conv2d(x,w1,[1,1,1,1], padding = "SAME")
	a1 = tf.nn.relu(z1)
	p1 = tf.nn.max_pool(a1, ksize = [1,2,2,1], strides = [1,1,1,1], padding = "SAME")

	fc1 = tf.contrib.layers.flatten(p1)
	fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs = 128, activation_fn = tf.nn.relu)
	fc3 = tf.contrib.layers.fully_connected(fc2, num_outputs = 24, activation_fn = None)

	return fc3

def loss(fc3, y):
	cost = tf.nn.softmax_cross_entropy_with_logits(logits = fc3, labels = y)
	cost = tf.reduce_mean(cost)
	return cost

def model(trainx, trainy, testx, testy, learning_rate = 0.009, num_epochs = 75):
	(m,nh0,nw0,nc0) = trainx.shape
	(m,ny) = trainy.shape
	seed = 2
	x,y = create_placeholders(nh0,nw0,nc0,ny)
	costs = []
	params = init_weights()
	fc3 = forward_prop(x, params)
	cost = loss(fc3, y)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			_, lcost = sess.run([optimizer, cost], feed_dict = {x:trainx, y:trainy})
			costs.append(lcost)
			print(epoch, lcost)
		ypred = tf.argmax(fc3, axis = 1)
		yreal = tf.argmax(y, axis = 1)
		correct = tf.equal(ypred, yreal)
		acc = tf.reduce_mean(tf.cast(correct, 'float'))
		train_acc = acc.eval({x:trainx, y:trainy})
		test_acc = acc.eval({x:testx, y:testy})
		print(train_acc, test_acc)
		return train_acc, test_acc, params
_,_, params = model(trainx, trainy, testx, testy)



