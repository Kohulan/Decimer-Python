'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by Kohulan Rajan
'''
#A simple implementation of image string reading and coverting them to numpy array
import numpy as np

fh = open("log.txt","r")
lines = fh.readlines()
strarray = []
for i in lines:
	strarray.append(i.strip().split(".0,[")[1])

fin = []
for i in strarray:
	str = i[0:-1]
	x = []
	x = list(map(float,str.split(',')))
	fin.append(x)
	
num_array = np.asarray(fin)

print(len(num_array[0]))