'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by Kohulan Rajan
'''
#Test code to check the data processing in parallel using multiprocessing

import numpy as np
from datetime import datetime
from numpy import array
import pickle
import lz4.frame as lz
import multiprocessing

Test_Images = pickle.load( open("File_test.txt","rb"))
test_items = Test_Images.items()
print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Loading done! Test",len(test_items))


def decomp(k):
	strarray_test = (lz.decompress(Test_Images.values()[k]))
	r = np.fromstring(strarray_test, dtype=float, sep=',')
	loadedImagest = np.array(r).astype(np.float32)
	test_array_enc=(1.0-loadedImagest/255.0)
	return test_array_enc


pool = multiprocessing.Pool(30)

result = pool.map(decomp,range(len(test_items)))

print (len(result[0]))