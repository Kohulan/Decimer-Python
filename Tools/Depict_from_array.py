'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by Kohulan Rajan
'''
#Implementation of matplot depiction, To depict an image from Image arrays
import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import array

with open('Image_array_binary.txt', 'r') as fp:
    strarray = []
    j = 0
    for i, line in enumerate(fp):
            strarray.append(line.strip().split(".0,[")[1])
            j+=1
            if j>= 5:
                    fin = []
                    for k in strarray:
                            str = k[0:-1]
                            x = 0
                            x = list(map(float,str.split(',')))
                            fin.append(x)
                    load = np.asarray(fin)
                    print (len(load))
                    break

print(len(load[0]))
print(load[0])
for i in range(len(load)):
	#sample = (0.95-(load[i]/255)).reshape(28,28)
	sample = (load[i]).reshape(512,512)
	plt.imshow(sample, cmap='Greys')
	plt.savefig('myfig_%i.png' % i, dpi=1000)