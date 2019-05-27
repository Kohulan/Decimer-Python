'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2019
'''
#Original source: https://github.com/nextmovesoftware/deepsmiles
#Deepsmiles decoding implementation for my work 

import sys
import numpy as np
import deepsmiles


print("DeepSMILES version: %s" % deepsmiles.__version__)
converter = deepsmiles.Converter(rings=True, branches=True)
print(converter) # record the options used

f = open('Output.txt','w',0)

with open("Input.txt","r") as fp:
	for i,line in enumerate(fp):
		id =(line.strip().split("\t")[1])
		smiles = (line.strip().split("\t")[0])
		
		try:
			decoded = converter.decode(smiles)
			f.wite(decoded+"\t\t"+id+"\n")
		except deepsmiles.DecodeError as e:
			decoded = None
			f.write(smiles+"DecodeError! Error message was '%s'" % e.message)

f.close()