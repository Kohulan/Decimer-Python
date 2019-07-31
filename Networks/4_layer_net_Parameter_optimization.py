'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by Kohulan Rajan
 * © 2019
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

#Set the Desired Gpu from the cluster
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Set Hidden neurons count
hidden_neurons_list_I = [2,4,8,16,32,64,128,512,1024,2048,4096]
hidden_neurons_list_II = [2,4,8,16,32,64,128,512,1024,2048,4096]

#Set Batch Size
batch_sizer_list = [500,1000]

#Set Learning rate
learning_rate_list = [0.001,0.003,0.005,0.007,0.009,0.01]

#Paramter Optimizing loops
for hidden_neurons_I in range(len(hidden_neurons_list_I)):
	for hidden_neurons_II in range(len(hidden_neurons_list_II)):
		for batch_sizer in range(len(batch_sizer_list)):
			for learning_rate_ in range(len(learning_rate_list)):
				f = open("/Results/Batch Size_{}_learning_rate_{}_hidden_neurons_{}_x_{}.txt".format(batch_sizer_list[batch_sizer],learning_rate_list[learning_rate_],hidden_neurons_list_I[hidden_neurons_I],hidden_neurons_list_II[hidden_neurons_II]), 'w',0)
				sys.stdout = f
				print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Network Started")

				#Data input from image data

				#labels
				def label_data(is_test=False):
					data_path = "train"
					if is_test:
						data_path = "test"
					myFile = open('/Data/Potential'+data_path+'_labels.csv',"r")
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
				Train_Images = pickle.load( open("/Data/train_compressed.txt","rb"))
				Test_Images = pickle.load( open("/Data/test_compressed.txt","rb"))
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
				training_epochs = 20
				batch_size = batch_sizer_list[batch_sizer]
				display_step = 1
				testbatch_size = 1000
				totaltrain_batch = len(train_items)/batch_size
				totaltest_batch = len(test_items)/testbatch_size

				# Network Parameters
				n_hidden_1 = hidden_neurons_list_I[hidden_neurons_I] # 1st layer number of neurons
				n_hidden_2 = hidden_neurons_list_II[hidden_neurons_II] # 1st layer number of neurons
				n_input = 256*256 # Data input (Image shape: 1024 * 1024)
				n_classes = 36 # Bond_Count

				# tf Graph input
				X = tf.placeholder("float", [None, n_input])
				Y = tf.placeholder("float", [None, n_classes])

				# Store layers weight & bias
				weights = {
					'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
					'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
					'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
				}
				biases = {
					'b1': tf.Variable(tf.random_normal([n_hidden_1])),
					'b2': tf.Variable(tf.random_normal([n_hidden_2])),
					'out': tf.Variable(tf.random_normal([n_classes]))
				}

				# Create model
				def multilayer_perceptron(x):
					# Fully Connected Hidden Layers
					layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
					layer_1 = tf.nn.relu(layer_1)
					
					layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
					layer_2 = tf.nn.relu(layer_2)

					# Output fully connected layer with a neuron for each class
					out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
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
				test_loss_history = []
				test_accuracy_history = []

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
						Train_loss_per_batch = 0
						
						# Loop over all batches
						for l in range(totaltrain_batch):
							print ("bathc",l)
							print ("tests","count",counter,"batchsize",counter+batch_size)
							train_batchX = pool_train.map(decomp_train,range(counter,counter+batch_size))
							batch_x=train_batchX
							batch_y=y_data_enc[counter:(counter+len(train_batchX))]
							_, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,Y: batch_y})
							Train_loss_per_batch += c 
							
				
							#Validation and calculating training accuracy
							_, accu_train = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
							valid_history.append(accu_train)
							total_correct_preds += accu_train
							print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"train Accuracy:",accu_train)
				
							print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),counter,"batch over")
							counter += len(train_batchX)
							
						validation_accuracy = total_correct_preds/totaltrain_batch
						print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Train accuracy:",validation_accuracy)
						acc_valid_history.append(validation_accuracy)
						loss_history.append(Train_loss_per_batch/totaltrain_batch)
						
						#Testing
						counter_test = 0
						All_test_loss = 0
						All_error = 0
						test_accuracy_perbatch = 0
						for test_set in range(totaltest_batch):
							X_test = pool_test.map(decomp_test,range(counter_test,counter_test+testbatch_size))
							Y_test = y_test_enc[counter_test:(counter_test+len(X_test))]
				
							test_acc = accuracy.eval({X: X_test, Y: Y_test})
							print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Accuracy:", test_acc)
							test_accuracy_perbatch += test_acc
							test_loss_batch,predict,error = sess.run([loss_op,pred,pred_difference], feed_dict={X: X_test, Y: Y_test})
							All_test_loss += test_loss_batch
							All_error += error
							#print(predict)
							counter_test += len(X_test)
								
				
							
						#Statistics	
						print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Final Test Accuracy:",test_accuracy_perbatch/totaltest_batch)	
						mean_error.append(np.absolute(np.mean(All_error/totaltest_batch)))	
						median_error.append(np.absolute(np.median(All_error/totaltest_batch)))	
						maximum_error.append(np.absolute(np.amax(All_error/totaltest_batch)))	
						test_loss_history.append(All_test_loss/totaltest_batch)	
						test_accuracy_history.append(test_accuracy_perbatch/totaltest_batch)	
							
						# Display logs per epoch step	
						if epoch % display_step == 0:	
							print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Epoch:", '%04d' % (epoch+1))	
					print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Optimization Finished!")	
					print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Network completed")	
					f.close()	
					pool_train.close()	
						
					
						
					
					# Final results for various bond counts
					file_append = open('/Results/Final_Report.txt' , 'a+')
					sys.stdout = file_append
					print("\n---------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
					print("Batch Size_{}_learning_rate_{}_hidden_neurons_{}_x_{}.txt".format(batch_sizer_list[batch_sizer],learning_rate_list[learning_rate_],hidden_neurons_list_I[hidden_neurons_I],hidden_neurons_list_II[hidden_neurons_II]))
					print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Final Train accuracy:",validation_accuracy)
					print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Final Test Accuracy:",test_accuracy_perbatch/totaltest_batch)
					counter_test_x = 0
					prediction_difference = 0
					for testing in range(totaltest_batch):
						X_test = pool_test.map(decomp_test,range(counter_test_x,counter_test_x+testbatch_size))
						Y_test = y_test_enc[counter_test_x:(counter_test_x+len(X_test))]
						_, predict,prediction_difference_batch = sess.run([loss_op,pred,pred_difference], feed_dict={X: X_test, Y: Y_test})
						prediction_difference += prediction_difference_batch
						counter_test_x += len(X_test)
						
					prediction_window = np.absolute(prediction_difference)
					pool_test.close()
					for j in range(10):
						count_error = 0
						for i in prediction_window:
							if i<=j: 
								count_error+=1
						Window_accuracy = float(count_error)/len(test_items)*100
						print("Currectly predicted bond count with error less than",j,"bonds, Accuracy ={:.2f}".format(Window_accuracy))
				file_append.close()
                                
                                #Matplot plot depiction
				plt.subplot(3,1,1)
				plt.plot(loss_history, '-o', label='Train Loss value')
				plt.title('Training & Tesing Loss')
				plt.xlabel('Epoch x Batches')
				plt.ylabel('Loss Value')
				plt.plot(test_loss_history, '-o', label='Test Loss value')
				plt.xlabel('Epoch x Batches')
				plt.ylabel('Loss Value')
				plt.legend(ncol=2, loc='upper right')
				plt.subplot(3,1,2)
				plt.gca().set_ylim([0,1.0])
				plt.plot(acc_valid_history, '-o', label='Train Accuracy value')
				plt.plot(test_accuracy_history, '-o', label='Test Accuracy value')
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
				plt.savefig("/Results/Batch Size_{}_learning_rate_{}_hidden_neurons_{}_x_{}.png".format(batch_sizer_list[batch_sizer],learning_rate_list[learning_rate_],hidden_neurons_list_I[hidden_neurons_I],hidden_neurons_list_II[hidden_neurons_II]))
				plt.close()