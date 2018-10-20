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
import pickle
import lz4.frame as lz

f = open('Report20180924.txt' , 'w',0)
sys.stdout = f

print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"packages loaded\n")

#Data input from image data
#labels
def label_data(is_test=False):
	data_path = "train"
	if is_test:
		data_path = "test"
	myFile = open('/media/data_drive/DECIMER_Test_Data/arrays/dataset/Labels_'+data_path+'.csv',"r")
	labels = []
	for row in myFile:
		x = int(row.strip().split(",")[1])
		labels.append(x)
	myFile.close()
	return np.asarray(labels)

y_train = label_data()
y_test = label_data(is_test=True)

print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Labels loaded !!")

#Image array data
Train_Images = pickle.load( open("train_compressed.txt","rb"))
Test_Images = pickle.load( open("test_compressed.txt","rb"))
train_items = Train_Images.items()
test_items = Test_Images.items()
print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Loading done! Train",len(train_items))
print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Loading done! Test",len(test_items))
#one hot vector transformation
def one_hot(y, n_labels):
	mat = np.zeros((len(y), n_labels))
	for i, val in enumerate(y):
		mat[i, val] = 1
	return mat

# Parameters
learning_rate = 0.005
training_epochs = 30
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
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#predicted labels vs original labels for statistics
pred = tf.subtract(tf.argmax(logits, 1), tf.argmax(Y, 1))

#Initiating data for plots
loss_history = []
acc_history = []
valid_history = []
acc_valid_history = []
difference_history = []
#totaltest_batch = int(len(y_test)/testbatch_size)
#user_test=1000
print ("All good!")


# Calculate accuracy
def test_array_build():
	counttest=0
	for l in range(1):
		strarray = []
		j=0 #Counter
		fint = []
		for i in range(len(test_items)):
			if i == counttest+j:
				strarray_test=lz.decompress(Test_Images.values()[i])
				#r = []
				r = np.fromstring(strarray, dtype=float, sep=',')
				fint.append(r)
				j+=1
				if j == testbatch_size:
					print ("Im Here!!")
					loadedImagest = np.array(fint).astype(np.float32)
					test_array_enc=(1.0-loadedImagest/255.0)
					counttest = j+counttest
					return test_array_enc
					break

temp = test_array_build()
print (len(temp))
#Network training
print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Training Started")
with tf.Session() as sess:
	sess.run(init)

	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(len(y_train)/batch_size)
		print ("total batch",total_batch)
		count=0
		total_correct_preds = 0

		# Loop over all batches
		for l in range(total_batch):
			print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),count,"batch over")
			strarray = []
			j=0 #Counter
			fin = []
			for i in range(len(train_items)):
				if i == count+j:
					strarray=lz.decompress(Train_Images.values()[i])
					#q = []
					q = np.fromstring(strarray, dtype=float, sep=',')
					fin.append(q)
					j+=1
					if j == batch_size:
						loadedImages = np.array(fin).astype(np.float32)
						batch_x=(1.0-loadedImages/255.0)
						#sample = (batch_x[0]).reshape(1024,1024)
						#plt.imshow(sample, cmap='Greys')
						#plt.savefig('Train_image.png', dpi=1000)
						batch_y=y_data_enc[count:(count+j)]
						_, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,Y: batch_y})
						avg_cost += c / total_batch
						loss_history.append(c)
						
						#Validation and calculating training accuracy
						_, accu_train = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
						valid_history.append(accu_train)
						total_correct_preds += accu_train
						print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"train Accuracy:",accu_train)
						
						#Testing
						X_test = temp
						Y_test = y_test_enc
						
						test_acc = accuracy.eval({X: X_test, Y: Y_test})
						print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Accuracy:", test_acc)
						acc_history.append(test_acc)
						difference_history.append(accu_train-test_acc)
						
						_, predict = sess.run([loss_op,pred], feed_dict={X: X_test, Y: Y_test})
						print (predict)
						
						count = j+count
						#print (count)
						break
		validation_accuracy = total_correct_preds/total_batch
		print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Train accuracy:",validation_accuracy)
		acc_valid_history.append(validation_accuracy)

		 # Dibeginay logs per epoch step
		if epoch % dibeginay_step == 0:
			print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
	print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Optimization Finished!")





f.close()

#Matplot plot depiction
plt.subplot(3,1,1)
plt.plot(loss_history, '-o', label='Loss value')
plt.title('Training Loss')
plt.xlabel('Epoch x Batches')
plt.ylabel('Loss Value')
plt.legend(ncol=2, loc='upper right')
plt.subplot(3,1,2)
plt.plot(valid_history, '-o', label='Train Accuracy value')
plt.plot(acc_history, '-o', label='Test Accuracy value')
plt.plot(difference_history, '-o', label='Train-Test Accuracy')
plt.title('Train & Test Accuracy')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
plt.subplot(3,1,3)
plt.plot(acc_valid_history, '-o', label='Final Train Accuracy value')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
plt.gcf().set_size_inches(15, 30)
plt.savefig('Plot_Final20180924.png')
plt.close()

