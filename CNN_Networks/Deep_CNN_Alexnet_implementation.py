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

f = open('Training_Report.txt' , 'w',0)
sys.stdout = f

print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"packages loaded\n")

#Data input from image data
#labels
def label_data(is_test=False):
	data_path = "train"
	if is_test:
		data_path = "test"
	myFile = open('Path/to/Labels'+data_path+'_labels.csv',"r")
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

#Training Parameters
learningrate = 0.0004
training_epochs = 20
batch_size = 512
display_step = 1
testbatch_size = len(test_items)
totaltrain_batch = len(train_items)/batch_size
totaltest_batch = len(test_items)/testbatch_size

# Network Parameters
num_input = (256*256) #Data input (img shape: 256*256)
num_classes = 36 # Total classes (36 digits)
dropout = 0.75 # Dropout, probability to keep units

# encoding labels to one_hot vectors
y_data_enc = one_hot(y_train, num_classes)
y_test_enc = one_hot(y_test, num_classes)

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])


# Wrappers
def conv2d(x, W, b, s=1,pad='VALID'): 
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=pad)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2, s=1,pad='VALID'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=pad)

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    # LRN wrapper
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)

def dropout(x, keepPro = 0.5, name = None):
    # Dropout wrapper
    return tf.nn.dropout(x, keepPro, name)

# Create model
def conv_net(x, weights, biases):
    # Data input is a 1-D vector of (256*256 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 256, 256, 1])

    # Convolution Layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'],s=4,pad='VALID')
    #Normalization
    lrn1 = LRN(conv1, 2, 2e-05, 0.75,'norm1')
    # Max Pooling (down-sampling) 1
    pool1 = maxpool2d(conv1,k=3,s=2,pad='VALID')#253
    
     # Convolution Layer 2
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'],s=1,pad='VALID')
    #Normalization
    lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
    # Max Pooling (down-sampling)
    pool2 = maxpool2d(conv2,k=3,s=1,pad='VALID')#22
    
    # Convolution Layer 3
    conv3 = conv2d(pool2, weights['wc3'], biases['bc3'],s=1,pad='SAME')
    
    # Convolution Layer 4
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'],s=1,pad='SAME')
    
    # Convolution Layer 5
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'],s=1,pad='SAME')
    # Max Pooling (down-sampling)
    pool3 = maxpool2d(conv5,k=2,s=2,pad='VALID')#22
   
    # Fully connected layer 1
    # Reshape conv5 output to fit fully connected layer input
    fc1 = tf.reshape(pool3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    # Apply Dropout
    dropout1 = dropout(fc1)
    
    # Fully connected layer 2
    fc2 = tf.add(tf.matmul(dropout1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    
    # Apply Dropout
    dropout2 = dropout(fc2)

    # Output, class prediction
    out = tf.add(tf.matmul(dropout2, weights['out']), biases['out'])
    out = tf.nn.softmax(out)
    return out

# Store layers weight & bias
weights = {
    # 12x12 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([11, 11, 1, 96], stddev=0.01)),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01)),
    'wc3': tf.Variable(tf.truncated_normal([3,3,256,384], stddev=0.01)),
    'wc4': tf.Variable(tf.truncated_normal([3,3,384,384], stddev=0.01)),
    'wc5': tf.Variable(tf.truncated_normal([3,3,384,256], stddev=0.01)),

    # fully connected, 7*7*32 inputs, 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([12*12*256, 4096], stddev=0.01)),
    'wd2': tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.01)),

    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([4096, num_classes], stddev=0.01))
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([96])),
    'bc2': tf.Variable(tf.truncated_normal([256])),
    'bc3': tf.Variable(tf.truncated_normal([384])),
    'bc4': tf.Variable(tf.truncated_normal([384])),
    'bc5': tf.Variable(tf.truncated_normal([256])),
    'bd1': tf.Variable(tf.truncated_normal([4096])),
    'bd2': tf.Variable(tf.truncated_normal([4096])),
    'out': tf.Variable(tf.truncated_normal([num_classes]))
}



# Construct model
logits_p = conv_net(X, weights, biases)
#prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_p, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningrate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(logits_p, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss_history = []
acc_history = []
valid_history = []
test_loss_history = []
test_accuracy_history = []



print ("All good!")

#-----------------------------------------------------------------------------------------------------------------
print ("Total available threads for multiprocessing: ",multiprocessing.cpu_count())

#Decompressing Lines Test
def decomp_test(k):
	strarraytest = (lz.decompress(Test_Images.values()[k]))
	floatarray_test = np.fromstring(strarraytest, dtype=float, sep=',')
	floatarray32_test = np.array(floatarray_test).astype(np.float32)
	#encoded_array_test=(1.0-floatarray32_test/255.0)
	return floatarray32_test

pool_test = multiprocessing.Pool()

def test_array_build():
	result = pool_test.map(decomp_test,range(testbatch_size))
	return result

temp = test_array_build()
pool_test.close()

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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# Start training
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
	sess.run(init)
	summary_writer = tf.summary.FileWriter('./Output', sess.graph)
	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0
		print ("total batch",totaltrain_batch)
		counter=0
		counterx = 0
		total_correct_preds = 0
		

		# Loop over all batches
		for o in range(totaltrain_batch):
			train_batchX = pool_train.map(decomp_train,range(counterx,counterx+batch_size))
			batch_x=train_batchX
			batch_y=y_data_enc[counterx:(counterx+len(train_batchX))]
			_,c = sess.run([train_op,loss_op], feed_dict={X: batch_x,Y: batch_y})
			counterx += len(train_batchX)
			
		for l in range(totaltrain_batch):
			batch_x=train_batchX
			batch_y=y_data_enc[counter:(counter+len(train_batchX))]
			loss,accu_train = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
			print("batch ",l,"  Batch Accuracy : ", accu_train)
			total_correct_preds += accu_train
			#print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"train Accuracy:",accu_train,"mini batch loss={:.2f}".format(loss))
			test_loss,test_accuracy_p = sess.run([loss_op, accuracy], feed_dict={X: temp[:512], Y: y_test_enc[:512]})
			test_accuracy_history.append(test_accuracy_p)
			test_loss_history.append(test_loss)
			loss_history.append(loss)
			acc_history.append(accu_train)
			

			counter += len(train_batchX)
		print("Iter " + str(l) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(accu_train))	
		print("Testing Accuracy:",sess.run(accuracy, feed_dict={X: temp[:512], Y: y_test_enc[:512]}))
		validation_accuracy = total_correct_preds/totaltrain_batch
		print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Train accuracy:",validation_accuracy)
		valid_history.append(validation_accuracy)		

		 # Display logs per epoch step
		if epoch % display_step == 0:
			print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
	print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Optimization Finished!")
	
	print("Testing Accuracy:",sess.run(accuracy, feed_dict={X: temp[:512], Y: y_test_enc[:512]}))
	#print (acc_history)
	summary_writer.close()
f.close()

#Matplot plot depiction
plt.subplot(2,1,1)
plt.plot(loss_history, '-o', label='Train Loss value')
plt.plot(test_loss_history, '-o',label="Test Loss Value" )
plt.title('Training Loss')
plt.xlabel('Epoch x Batches')
plt.ylabel('Loss Value')
plt.legend(ncol=2, loc='upper right')
plt.subplot(2,1,2)
plt.gca().set_ylim([0,1.0])
plt.plot(acc_history, '-o', label='Train Accuracy value')
plt.plot(test_accuracy_history,'-o',label = 'Test Accuracy Value')
#plt.plot(difference_history, '-o', label='Train Accuracy')
plt.title('Train Accuracy')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
plt.gcf().set_size_inches(15, 30)
plt.savefig('Plot.png')
plt.close()