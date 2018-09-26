<<<<<<< HEAD
'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by Kohulan Rajan
'''
import sys
import numpy as np
import lz4.frame as lz
from guppy import hpy
import pickle
from collections import OrderedDict

h = hpy()
images= OrderedDict()


with open("test.txt","r") as fp:
	for i,line in enumerate(fp):
		#print ("hi")
		chembl =(line.strip().split(".0,[")[0])
		str = (line.strip().split(".0,[")[1])
		imagestr =(lz.compress(str[0:-1]))
		images[chembl]=imagestr
		#if i / 1000.0 == int(i/1000.0):
		#	print (h.heap())
print (sys.getsizeof(images))

pickle.dump(images,open("test_compressed.txt","wb"))
=======
'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by Kohulan Rajan
'''
import sys
import numpy as np
import lz4.frame as lz
from guppy import hpy
import pickle
from collections import OrderedDict

h = hpy()
images= OrderedDict()


with open("test.txt","r") as fp:
	for i,line in enumerate(fp):
		#print ("hi")
		chembl =(line.strip().split(".0,[")[0])
		str = (line.strip().split(".0,[")[1])
		imagestr =(lz.compress(str[0:-1]))
		images[chembl]=imagestr
		#if i / 1000.0 == int(i/1000.0):
		#	print (h.heap())
print (sys.getsizeof(images))

pickle.dump(images,open("test_compressed.txt","wb"))
>>>>>>> origin
