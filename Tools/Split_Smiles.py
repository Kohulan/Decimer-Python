'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2019
'''
#Splits the smiles string into individual characters

input_file = open('Input.txt','r')


with open('Output.txt', 'a') as the_file:
	for i,line in enumerate(input_file):
		chembl =(line.strip().split(",")[0])
		new_smile =(line.strip().split(",")[1])
		new_smiles = list(new_smile)
		the_file.write(chembl)
		the_file.write(",")
		for item in new_smiles:
			the_file.write("%s " % item)
		#the_file.write(new_smile)
		the_file.write("\n")	

print("Smiles splitted!")
