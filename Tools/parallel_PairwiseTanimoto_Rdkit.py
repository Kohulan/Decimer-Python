'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2019
'''

#Parallel Pairwise Tanimoto Calculation, Serial code took  3m 33.132s to process 500 x 500 Pairwise Tanimoto similarity
#calculation, Using Multiprocess We could achieve the same processing in 14.496s, almost 15 fold reduction in time.

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import sys
import numpy as np
import multiprocessing as mp
f = open('Output.txt','w')

smiles = []

with open("Input.txt","r") as fp:
	for i,line in enumerate(fp):
		smi =(line.strip().split("\t\t")[0])
		#Id = (line.strip().split("\t\t")[1:]) #Optional
		smiles.append(smi)

def cal_pairwise_tanimoto(pair):
	i,j = (pair)
	x = Chem.MolFromSmiles(smiles[i])
	y = Chem.MolFromSmiles(smiles[j])
	fps1 = FingerprintMols.FingerprintMol(x)
	fps2 = FingerprintMols.FingerprintMol(y)
	tani = DataStructs.TanimotoSimilarity(fps1,fps2)
	values = (str(i)+","+str(j)+","+str(tani)+"\n")
	return values
	
if __name__ == '__main__':
	pool = mp.Pool(40)
	for i in range(len(smiles)):
		arg_pairs = [(i,j) for j in range(i+1,len(smiles))]
		#print (i,j)
		return_values = pool.map(cal_pairwise_tanimoto,arg_pairs)
		for i in range(len(return_values)):
			f.write(return_values[i])
	pool.close()	
	f.close()
