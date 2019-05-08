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
import multiprocessing

f = open('CNN_reportv100.txt' , 'w',0)
sys.stdout = f

print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"packages loaded\n")

#Data input from image data
#labels
def label_data(is_test=False):
	data_path = "train"
	if is_test:
		data_path = "test"
	myFile = open('Potential'+data_path+'_labels.csv',"r")
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
Train_Images = pickle.load( open("Potential_train_compressed.txt","rb"))
Test_Images = pickle.load( open("Potential_test_compressed.txt","rb"))
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

#Training Parameters
learningrate = 0.004
epochs = 2
batch_size = 128
display_step = 1
testbatch_size = 128
totaltrain_batch = 100#len(train_items)/batch_size
totaltest_batch = len(test_items)/testbatch_size

# Network Parameters
num_input = (256*256) # MNIST data input (img shape: 28*28)
num_classes = 133 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
Fully_Connected_1 = 4096
Fully_Connected_2 = 4096

#Resetting graph
tf.reset_default_graph()

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


new = tf.reshape(X, shape=[-1, 256, 256, 1])
##CONVOLUTION LAYER 1
#Weights for layer 1
w_1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.01))
#Bias for layer 1
b_1 = tf.Variable(tf.constant(0.0, shape=[[5,5,1,32][3]]))
#Applying convolution
c_1 = tf.nn.conv2d(new, w_1,strides=[1, 2, 2, 1], padding='VALID')
#Adding bias
c_1 = c_1 + b_1
#Applying RELU
c_1 = tf.nn.relu(c_1)
								
print(c_1)

##CONVOLUTION LAYER 2
#Weights for layer 2
w_2 = tf.Variable(tf.truncated_normal([5,5,32,32], stddev=0.01))
#Bias for layer 2
b_2 = tf.Variable(tf.constant(0.0, shape=[[5,5,32,32][3]]))
#Applying convolution
c_2 = tf.nn.conv2d(c_1, w_2,strides=[1, 1, 1, 1], padding='VALID')
#Adding bias
c_2 = c_2 + b_2
#Applying RELU
c_2 = tf.nn.relu(c_2)
								
print(c_2)

##CONVOLUTION LAYER 3
#Weights for layer 3
w_3 = tf.Variable(tf.truncated_normal([5,5,32,32], stddev=0.01))
#Bias for layer 3
b_3 = tf.Variable(tf.constant(0.0, shape=[[5,5,32,32][3]]))
#Applying convolution
c_3 = tf.nn.conv2d(c_2, w_3,strides=[1, 1, 1, 1], padding='VALID')
#Adding bias
c_3 = c_3 + b_3
#Applying RELU
c_3 = tf.nn.relu(c_3)
								
print(c_3)

##POOLING LAYER1
p_1 = tf.nn.max_pool(c_3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
print(p_1)

##CONVOLUTION LAYER 4
#Weights for layer 4
w_4 = tf.Variable(tf.truncated_normal([5,5,32,32], stddev=0.01))
#Bias for layer 4
b_4 = tf.Variable(tf.constant(1.0, shape=[[5,5,32,32][3]]))
#Applying convolution
c_4 = tf.nn.conv2d(p_1, w_4,strides=[1, 1, 1, 1], padding='SAME')
#Adding bias
c_4 = c_4 + b_4
#Applying RELU
c_4 = tf.nn.relu(c_4)

print(c_4)

##POOLING LAYER2
p_2 = tf.nn.max_pool(c_4, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
print(p_2)

##CONVOLUTION LAYER 5
#Weights for layer 5
w_5 = tf.Variable(tf.truncated_normal([5,5,32,128], stddev=0.01))
#Bias for layer 4
b_5 = tf.Variable(tf.constant(1.0, shape=[[5,5,32,128][3]]))
#Applying convolution
c_5 = tf.nn.conv2d(p_2, w_5,strides=[1, 1, 1, 1], padding='SAME')
#Adding bias
c_5 = c_5 + b_5
#Applying RELU
c_5 = tf.nn.relu(c_5)

print(c_5)

##POOLING LAYER3
p_3 = tf.nn.max_pool(c_5, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
print(p_3)

##CONVOLUTION LAYER 5
#Weights for layer 5
w_5 = tf.Variable(tf.truncated_normal([5,5,128,512], stddev=0.01))
#Bias for layer 5
b_5 = tf.Variable(tf.constant(1.0, shape=[[5,5,128,512][3]]))
#Applying convolution
c_5 = tf.nn.conv2d(p_3, w_5,strides=[1, 1, 1, 1], padding='SAME')
#Adding bias
c_5 = c_5 + b_5
#Applying RELU
c_5 = tf.nn.relu(c_5)

print(c_5)

##POOLING LAYER 5

p_4 = tf.nn.max_pool(c_5, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
print(p_4)

#Flattening
flattened = tf.reshape(p_4,[-1,7*7*512])
print(flattened)


##Fully Connected Layer 1
flattened_input = int( flattened.get_shape()[1] )
#Weights for FC Layer 1
w1_fc = tf.Variable(tf.truncated_normal([flattened_input,Fully_Connected_1], stddev=0.01))
#Bias for FC Layer 1
b1_fc = tf.Variable( tf.constant(1.0, shape=[Fully_Connected_1] ) )
#Summing Matrix calculations and bias
sum_fc1 = tf.matmul(flattened, w1_fc) + b1_fc
#Applying RELU
sum_fc1 = tf.nn.relu(sum_fc1)
print(sum_fc1)

##Fully Connected Layer 2

#Weights for FC Layer 2
w2_fc = tf.Variable(tf.truncated_normal([Fully_Connected_1,Fully_Connected_2], stddev=0.01))
#Bias for FC Layer 1
b2_fc = tf.Variable( tf.constant(1.0, shape=[Fully_Connected_2] ) )
#Summing Matrix calculations and bias
sum_fc2 = tf.matmul(sum_fc1, w2_fc) + b2_fc
#Applying RELU
sum_fc2 = tf.nn.relu(sum_fc2)
print(sum_fc1)

##Fully Connected Layer 3
#Weights for FC Layer 3
w3_fc = tf.Variable(tf.truncated_normal([Fully_Connected_2,num_classes], stddev=0.01))
#Bias for FC Layer 3b3_fc = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
b3_fc = tf.Variable( tf.constant(1.0, shape=[num_classes] ) )
#Summing Matrix calculations and bias
y_pred = tf.matmul(sum_fc2, w3_fc) + b3_fc
#Applying RELU
print(y_pred)

#Defining loss function
prediction = tf.nn.softmax(y_pred)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=y_pred))

#Defining objective
train = tf.train.AdamOptimizer(learning_rate=learningrate).minimize(cross_entropy)

#Defining Accuracy
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# encoding labels to one_hot vectors
y_data_enc = one_hot(y_train, num_classes)
y_test_enc = one_hot(y_test, num_classes)

#Initializing weights
init = tf.global_variables_initializer()

#Initiating data for plots
loss_history = []
acc_history = []
valid_history = []
acc_valid_history = []
acc_test_history = []

print ("All good!")

#-----------------------------------------------------------------------------------------------------------------
print ("Total available threads for multiprocessing: ",multiprocessing.cpu_count())

#Decompressing Lines Test
def decomp_test_final(m):
	strarraytest = (lz.decompress(Test_Images.values()[m]))
	floatarray_test = np.fromstring(strarraytest, dtype=float, sep=',')
	floatarray32_test = np.array(floatarray_test).astype(np.float32)
	#encoded_array_test=(1.0-floatarray32_test/255.0)
	return floatarray32_test

pool_test_final = multiprocessing.Pool()

def test_array_build_final():
	result = pool_test_final.map(decomp_test_final,range(len(test_items)))
	return result

temp_final = test_array_build_final()
pool_test_final.close()

print ("Total test : ",len(temp_final))

def decomp_train(j):
	strarray = (lz.decompress(Train_Images.values()[j]))
	floatarray = np.fromstring(strarray, dtype=float, sep=',')
	floatarray32 = np.array(floatarray).astype(np.float32)
	#encoded_array=(1.0-floatarray32/255.0)
	return floatarray32

pool_train = multiprocessing.Pool()

#Network training
print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Training Started")

#GPU settings
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
# Start training
with tf.Session(config=config) as sess:
	sess.run(init)
	# Training cycle
	for epoch in range(epochs):
		avg_cost = 0
		print ("total batch",totaltrain_batch)
		counter=0
		total_correct_preds = 0

		# Loop over all batches
		for l in range(totaltrain_batch):
			print ("batch",l)
			print ("tests","count",counter,"batchsize",counter+batch_size)
			train_batchX = pool_train.map(decomp_train,range(counter,counter+batch_size))
			batch_x=train_batchX
			batch_y=y_data_enc[counter:(counter+len(train_batchX))]
			
			_,c = sess.run([train,cross_entropy], feed_dict={X: batch_x,Y: batch_y})
			accu_train,loss = sess.run([acc,cross_entropy], feed_dict={X: batch_x,Y: batch_y})
			loss_history.append(loss)
			
			#Validation and calculating training accuracy
			valid_history.append(accu_train)
			total_correct_preds += accu_train
			print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Loss={:.2f}".format(loss),"Mini batch accuracy:",accu_train)
			counter += len(train_batchX)
			
		validation_accuracy = total_correct_preds/totaltrain_batch
		print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Train accuracy:",validation_accuracy)
		Test_accuracy = sess.run(acc, feed_dict={X: temp_final[0:128], Y: y_test_enc[0:128]})
		print("Testing Accuracy:",Test_accuracy)
		acc_test_history.append(Test_accuracy)
		acc_valid_history.append(validation_accuracy)

		 # Dibeginay logs per epoch step
		#if epoch % display_step == 1:
		#	print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
	print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Optimization Finished!")
	
	print("Testing Accuracy:",sess.run(acc, feed_dict={X: temp_final[0:256], Y: y_test_enc[0:256]}))
	#print (acc_history)
f.close()
pool_train.close()

#Matplot plot depiction
plt.subplot(3,1,1)
plt.plot(loss_history, '-o', label='Loss value')
plt.title('Training Loss')
plt.xlabel('Epoch x Batches')
plt.ylabel('Loss Value')
plt.legend(ncol=2, loc='upper right')
plt.subplot(3,1,2)
plt.gca().set_ylim([0,1.0])
plt.plot(valid_history, '-o', label='Mini batch accuracy value')
#plt.plot(difference_history, '-o', label='Train-Test Accuracy')
plt.title('Mini Batch accuracy')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
plt.subplot(3,1,3)
plt.gca().set_ylim([0,1.0])
plt.title('Train & Test Accuracy')
plt.plot(acc_valid_history, '-o', label='Train Accuracy value')
plt.plot(acc_test_history, '-o', label='Test Accuracy value')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
plt.gcf().set_size_inches(15, 30)
plt.savefig('CNN_reportv10020190329.png')
plt.close()
