'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by Kohulan Rajan
'''
#Using multiprocessing for data reading in parallel and spliting them into batches for training.

import numpy as np
from datetime import datetime
from numpy import array
import pickle
import lz4.frame as lz
import multiprocessing

begin_time = datetime.now()
Test_Images = pickle.load( open("small_test_doc.txt","rb"))
Train_Images = pickle.load( open("small_test_doc.txt","rb"))
train_items = Train_Images.items()
test_items = Test_Images.items()
print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Loading done! Test",len(test_items))

training_epochs = 1
batchsize = 500
totaltest_batch = len(test_items)/batchsize
print(totaltest_batch)

def decomp_test(k):
	strarray_test = (lz.decompress(Test_Images.values()[k]))
	r = np.fromstring(strarray_test, dtype=float, sep=',')
	loadedImagest = np.array(r).astype(np.float32)
	test_array_enc=(1.0-loadedImagest/255.0)
	return test_array_enc

def decomp_train(j):
	strarray_test = (lz.decompress(Train_Images.values()[j]))
	r = np.fromstring(strarray_test, dtype=float, sep=',')
	loadedImagest = np.array(r).astype(np.float32)
	test_array_enc=(1.0-loadedImagest/255.0)
	return loadedImagest

pool_test = multiprocessing.Pool()

def testbatch():
	result = pool_test.map(decomp_test,range(len(test_items)))
	print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"batch",len(result),"type",type(result))
	return result


temp = testbatch()


print("succes of test",len(temp))

pool_train = multiprocessing.Pool()

for epoch in range(training_epochs):
	counter = 0
	for l in range(totaltest_batch):
		print ("batch",l)
		print ("tests","count",counter,"batchsize",counter+batchsize)
		result = pool_train.map(decomp_train,range(counter,counter+batchsize))
		counter += len(result)
		print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"batch",len(result),"type",type(result))
		print ("Batch over",counter)

print (len(result[0]))

print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Network completed")