'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2019
'''
#Using Rdkit libraries, calculates the Tanimoto similarity of Original and predicted Smiles and When a Predicted Smiles cannot be 
#used for calculation the expection is noted and printed on the report

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Original Vs Predicted SMILES, Tanimoto similarity check")
# Input Arguments
parser.add_argument(
	'--input',
	help = 'Enter the input filename',
	required = True
	)
parser.add_argument(
	'--output',
	help = 'Enter the output filename as desired',
	required = True
)
args= parser.parse_args()

f = open(args.output,'w')

#sys.stdout = f

smiles = []

with open(args.input,"r") as fp:
	for i,line in enumerate(fp):
		smi =(line.strip().split("\t\t")[0])
		Type = (line.strip().split("\t\t")[1])
		smiles.append(smi)

ms =[]

for k in range(0,len(smiles),2):
	#print ("My counter ",  k)
	try:
		x = Chem.MolFromSmiles(smiles[k])
		y = Chem.MolFromSmiles(smiles[k+1])
		#print ("Error", smiles[i+1],e)
		fps1 = FingerprintMols.FingerprintMol(x)
		fps2 = FingerprintMols.FingerprintMol(y)
		tani = DataStructs.TanimotoSimilarity(fps1,fps2)
		f.write(smiles[k]+"   Original Smiles\t")
		f.write(smiles[k+1]+"   Predicted Smiles\t")
		f.write("Tanimoto Smilarity :\t"+ str(tani)+"\n")
		#print("Original : ",k,"Predicted : ",k+1," Tanimoto Smilarity : ",tani)
	except Exception as e:
		f.write(smiles[k]+"   Original Smiles\t")
		f.write(smiles[k+1]+"   Predicted Smiles\t")
		f.write("Smiles String rejected\n")
		continue




f.close()