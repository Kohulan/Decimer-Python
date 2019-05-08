'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by Kohulan Rajan
'''
#Parallelized datareading network

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
np.set_printoptions(threshold=np.nan)

hidden_neurons_list = [2,4,8,16,32,64,128,512,1024,2048,4096]
batch_sizer_list = [256,512,1024]
learning_rate_list = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
#Paramter Optimizing loops
for hidden_neurons in range(len(hidden_neurons_list)):
	for batch_sizer in range(len(batch_sizer_list)):
		for learning_rate_ in range(len(learning_rate_list)):
			f = open("3Layer_Reports_201904/Batch Size_{}_learning_rate_{}_hidden_neurons_{}_.txt".format(batch_sizer_list[batch_sizer],learning_rate_list[learning_rate_],hidden_neurons_list[hidden_neurons]), 'w',0)
			sys.stdout = f
			print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Network Started")
			
			#Data input from image data

			#labels
			def label_data(is_test=False):
				data_path = "train"
				if is_test:
					data_path = "test"
				myFile = open('My'+data_path+'_labels.csv',"r")
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
			Train_Images = pickle.load( open("My_train_compressed.txt","rb"))
			Test_Images = pickle.load( open("My_test_compressed.txt","rb"))
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
			learning_rate = learning_rate_list[learning_rate_]
			training_epochs = 10
			batch_size = batch_sizer_list[batch_sizer]
			display_step = 1
			testbatch_size = len(test_items)
			totaltrain_batch = len(train_items)/batch_size
			totaltest_batch = len(test_items)/testbatch_size

			# Network Parameters
			n_hidden_1 = hidden_neurons_list[hidden_neurons] # 1st layer number of neurons
			n_input = 256*256 # Data input (Image shape: 256 * 256)
			n_classes = 36 # Bond_Count

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
				# Hidden fully connected layer with 384 neurons
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
			difference_history = []

			print ("Data decompression for test batch started!")

			#-----------------------------------------------------------------------------------------------------------------
			print ("Total available threads for multiprocessing: ",multiprocessing.cpu_count())

			#Decompressing Lines Test
			def decomp_test(k):
				strarraytest = (lz.decompress(Test_Images.values()[k]))
				floatarray_test = np.fromstring(strarraytest, dtype=float, sep=',')
				floatarray32_test = np.array(floatarray_test).astype(np.float32)
				encoded_array_test=(1.0-floatarray32_test/255.0)
				return encoded_array_test

			pool_test = multiprocessing.Pool()

			def test_array_build():
				result = pool_test.map(decomp_test,range(len(test_items)))
				return result

			temp = test_array_build()
			print (len(temp))

			def decomp_train(j):
				strarray = (lz.decompress(Train_Images.values()[j]))
				floatarray = np.fromstring(strarray, dtype=float, sep=',')
				floatarray32 = np.array(floatarray).astype(np.float32)
				encoded_array=(1.0-floatarray32/255.0)
				return encoded_array

			pool_train = multiprocessing.Pool()

			#Network training
			print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Training Started")
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True
			config.gpu_options.allocator_type = 'BFC'
			
			with tf.Session(config=config) as sess:
				sess.run(init)

				# Training cycle
				for epoch in range(training_epochs):
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
						_, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,Y: batch_y})
						avg_cost += c / totaltrain_batch
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
						_, predict,error = sess.run([loss_op,pred,pred_difference], feed_dict={X: X_test, Y: Y_test})

						print(predict)
						mean_error.append(np.absolute(np.mean(error)))
						median_error.append(np.absolute(np.median(error)))
						maximum_error.append(np.absolute(np.amax(error)))
						print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),counter,"batch over")
						counter += len(train_batchX)

					validation_accuracy = total_correct_preds/totaltrain_batch
					print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Train accuracy:",validation_accuracy)
					acc_valid_history.append(validation_accuracy)

					 # Display logs per epoch step
					if epoch % display_step == 0:
						print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
				print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Optimization Finished!")
				#print (acc_history)



			print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Network completed")
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
			plt.plot(valid_history, '-o', label='Train Accuracy value')
			plt.plot(acc_history, '-o', label='Test Accuracy value')
			#plt.plot(difference_history, '-o', label='Train-Test Accuracy')
			plt.title('Train & Test Accuracy')
			plt.xlabel('Batches')
			plt.ylabel('Accuracy')
			plt.legend(ncol=2, loc='lower right')
			plt.subplot(3,1,3)
			plt.plot(mean_error, '-o', label='Mean of error')
			plt.plot(median_error, '-o', label='Median of error')
			plt.plot(maximum_error, '-o', label='Maximum error')
			plt.xlabel('Batches')
			plt.ylabel('Error')
			plt.legend(ncol=2, loc='lower right')
			plt.gcf().set_size_inches(15, 30)
			plt.savefig("3Layer_Reports_201904/Batch Size_{}_learning_rate_{}_hidden_neurons_{}_.jpg".format(batch_sizer_list[batch_sizer],learning_rate_list[learning_rate_],hidden_neurons_list[hidden_neurons]))
			plt.close()