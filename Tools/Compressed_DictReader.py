'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by Kohulan Rajan
 
#This code is written to check how well the decompression algorithm performs inside the main network code.
'''

import numpy as np
import lz4.frame as lz
import pickle
from datetime import datetime
from itertools import islice

Images = pickle.load( open("compressed.txt","rb"))
print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Loading done!",len(Images))

#fl= open("uncompressed.txt","w", 0)
n_items = Images.items()
'''
print (len(n_items))
fin = []
for i in range(len(n_items)):
	#print(key[0])
	strarray=lz.decompress(Images.values()[i])
	#print(Images.keys()[i],lz.decompress(Images.values()[i]))
	#fl.write(strarray)
	x = []
	x = list(map(float,strarray.split(',')))
	fin.append(x)
	
num_array = np.asarray(fin)

print(len(num_array))
print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Uncompression Completed",len(num_array))
fl.close()
'''
batch_size = 2
total_batch = 4
count = 0
for l in range(total_batch+1):
	print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),count,"batch over")
	strarray = []
	j=0 #Counter
	fin = []
	for i in range(len(n_items)):
		if i == count+j:
			strarray=lz.decompress(Images.values()[i])
			q = []
			q = list(map(float,strarray.split(',')))
			fin.append(q)
			j+=1
			if j == batch_size:
				loadedImages = np.array(fin).astype(np.float32)
				count = j+count
				break

print ("Done")