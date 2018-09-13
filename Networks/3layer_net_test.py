'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by Kohulan Rajan
'''
#The first fully working 3 layer network
import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib as mpl
import csv
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import array


f = open('Diversereport20180907.txt' , 'w',0)
sys.stdout = f

print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"packages loaded\n")

#Data input from image data
def label_data(is_test=False):
	data_path = "train"
	if is_test:
		data_path = "test"
	myFile = open('/home/kohulan/Data_test/network_test_dataset/Labels_'+data_path+'.csv',"r")
	labels = []
	for row in myFile:
		x = int(row.strip().split(",")[1])
		labels.append(x)
	myFile.close()
	return np.asarray(labels)

y_train = label_data()
y_test = label_data(is_test=True)

print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Labels loaded !!")

#one hot vector transformation
def one_hot(y, n_labels):
	mat = np.zeros((len(y), n_labels))
	for i, val in enumerate(y):
		mat[i, val] = 1
	return mat

# Parameters
learning_rate = 0.1
training_epochs = 10
batch_size = 1000
dibeginay_step = 1
testbatch_size = 1000

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_input = 1024*1024 # Data input (Image shape: 1024 * 1024)
n_classes = 132 # Bond_Count

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
	# Hidden fully connected layer with 256 neurons
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	
	# Output fully connected layer with a neuron for each class
	out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
	return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()

# encoding labels to one_hot vectors
y_data_enc = one_hot(y_train, n_classes)
y_test_enc = one_hot(y_test, n_classes)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pd = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy_train = tf.reduce_mean(tf.cast(correct_pd, tf.float32))

#Initiating data for plots
loss_history = []
acc_history = []
acc_valid_history = []

print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Training Started")
with tf.Session() as sess:
	sess.run(init)

	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(len(y_train)/batch_size)
		count=0
		counttest=0
		total_correct_preds = 0

		# Loop over all batches
		for l in range(total_batch+1):
			print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),count,"batch over")
			with open('/home/kohulan/Data_test/network_test_dataset/train.txt', 'r') as fp:
				strarray = []
				j=0 #Counter
				for i, line in enumerate(fp):
					if i == count+j:
						strarray.append(line.strip().split(".0,[")[1])
						j+=1
						if j >= batch_size:
							fin = []
							for k in strarray:
								str = k[0:-1]
								q = []
								q = list(map(float,str.split(',')))
								fin.append(q)
							loadedImages = np.array(fin).astype(np.float32)
							batch_x=(1.0-loadedImages/255.0)
							sample = (loadedImages[0]).reshape(1024,1024)
							plt.imshow(sample, cmap='Greys')
							plt.savefig('Train_image.png', dpi=1000)
							batch_y=y_data_enc[count:(count+j)]
							_, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,Y: batch_y})
							avg_cost += c / total_batch
							loss_history.append(c)
							#Validation and calculating training accuracy
							_, accu_train = sess.run([loss_op, accuracy_train], feed_dict={X: batch_x,Y: batch_y})
							total_correct_preds += accu_train
							count = j+count
							break
		validation_accuracy = total_correct_preds/total_batch
		print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Train accuracy:",validation_accuracy)
		acc_valid_history.append(validation_accuracy)

		 # Dibeginay logs per epoch step
		if epoch % dibeginay_step == 0:
			print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
	print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Optimization Finished!")

	# Test model
	pred = tf.nn.softmax(logits)  # Apply softmax to logits
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	totaltest_batch = int(len(y_test)/testbatch_size)
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	for q in range(totaltest_batch):
		print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),counttest,"batch over")
		with open('/home/kohulan/Data_test/network_test_dataset/test.txt', 'r') as fpt:
			strarrayt = []
			e=0
			for w, linet in enumerate(fpt):
				if w == counttest+e:
					strarrayt.append(linet.strip().split(".0,[")[1])
					e+=1
					if e >= testbatch_size:
						fint = []
						for m in strarrayt:
							str = k[0:-1]
							r = []
							r = list(map(float,str.split(',')))
							fint.append(r)
						loadedImagest = np.array(fint).astype(np.float32)
						X_test=(1.0-loadedImagest/255.0)
						sample = (loadedImagest[0]).reshape(1024,1024)
						plt.imshow(sample, cmap='Greys')
						plt.savefig('Test_image.png', dpi=1000)
						Y_test=y_test_enc[counttest:(counttest+e)]
						print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
						acc_history.append(accuracy.eval({X: X_test, Y: Y_test}))
						counttest = e+counttest
						break

f.close()

#Matplot plot depiction
plt.subplot(3,1,1)
plt.plot(loss_history, '-o', label='Loss value')
plt.title('Training Loss')
plt.xlabel('Epoch x Batches')
plt.ylabel('Loss Value')
plt.legend(ncol=2, loc='upper right')
plt.subplot(3,1,2)
plt.plot(acc_valid_history, '-o', label='Train Accuracy value')
plt.title('Train Accuracy')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
plt.subplot(3,1,3)
plt.plot(acc_history, '-o', label='Test Accuracy value')
plt.title('Test Accuracy')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
plt.gcf().set_size_inches(15, 30)
plt.savefig('Plot_Final20180906.png')
plt.close()


