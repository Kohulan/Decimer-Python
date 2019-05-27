'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2019
'''

#Used to calculate Pairwise Tanimoto Similarity for a given set of smiles on a document. The parrallized code also available in the sam repository

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import sys
import numpy as np

f = open('Output.txt','w')

smiles = []

with open("Input.txt","r") as fp:
	for i,line in enumerate(fp):
		smi =(line.strip().split("\t\t")[0])
		#Type = (line.strip().split("\t\t")[0])
		smiles.append(smi)

for k in range(len(smiles)):
	for j in range(k+1,len(smiles)):
		#print(j)
		x = Chem.MolFromSmiles(smiles[k])
		y = Chem.MolFromSmiles(smiles[j])
		fps1 = FingerprintMols.FingerprintMol(x)
		fps2 = FingerprintMols.FingerprintMol(y)
		tani = DataStructs.TanimotoSimilarity(fps1,fps2)
		values = (k,j,tani)
		f.write(str(k))
		f.write(",")
		f.write(str(j))
		f.write(",")
		f.write(str(tani))
		f.write("\n")

f.close()