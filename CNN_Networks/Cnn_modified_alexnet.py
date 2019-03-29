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
learning_rate = 0.004
epochs = 2
batch_size = 128
display_step = 1
testbatch_size = 128
totaltrain_batch = len(train_items)/batch_size
totaltest_batch = len(test_items)/testbatch_size

# Network Parameters
num_input = (256*256) # MNIST data input (img shape: 28*28)
num_classes = 133 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=2): 
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=0,strides = 0):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 256, 256, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'],strides=2)#126 with filter of 12x12
    print (tf.shape(conv1,out_type=tf.int32))
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1,k=2,strides = 1)#42
    
     # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'],strides=2)#20 with filter of 4x4
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2,k=2,strides = 1)#10 

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 12x12 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 96])),
    # fully connected, 32*32*128 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([64*64*96, 1024])),
    # 1024 inputs, 133 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([96])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# encoding labels to one_hot vectors
y_data_enc = one_hot(y_train, num_classes)
y_test_enc = one_hot(y_test, num_classes)

# Evaluate the errors, mean,median and maximum errors
pred = tf.argmax(logits, 1)
pred_difference = tf.subtract(tf.argmax(Y, 1),tf.argmax(logits, 1))
mean_error=[]
median_error=[]
maximum_error=[]

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
			
			sess.run(train_op, feed_dict={X: batch_x,Y: batch_y, keep_prob: dropout})
			loss,accu_train = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y,keep_prob: 1.0})
			loss_history.append(loss)
			
			#Validation and calculating training accuracy
			valid_history.append(accu_train)
			total_correct_preds += accu_train
			print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Loss={:.2f}".format(loss),"Mini batch accuracy:",accu_train)
			counter += len(train_batchX)
			
		validation_accuracy = total_correct_preds/totaltrain_batch
		print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Train accuracy:",validation_accuracy)
		Test_accuracy = sess.run(accuracy, feed_dict={X: temp_final[0:128], Y: y_test_enc[0:128], keep_prob: 1.0})
		print("Testing Accuracy:",Test_accuracy)
		acc_test_history.append(Test_accuracy)
		acc_valid_history.append(validation_accuracy)

		 # Dibeginay logs per epoch step
		#if epoch % display_step == 1:
		#	print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
	print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Optimization Finished!")
	
	print("Testing Accuracy:",sess.run(accuracy, feed_dict={X: temp_final[0:256], Y: y_test_enc[0:256], keep_prob: 1.0}))
	#print (acc_history)
f.close()

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
plt.savefig('CNN_report1.png')
plt.close()
