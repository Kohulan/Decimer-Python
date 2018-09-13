'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by : Kohulan.R on 2018/08/04
'''
#Implementation of diverse molecule picking using RDKit

from rdkit import Chem
from rdkit.Chem import Draw,rdMolDescriptors,AllChem
from rdkit import SimDivFilters,DataStructs
import gzip, time, platform
from datetime import datetime

#Timestamp
datetime.now().strftime('%Y/%m/%d %H:%M:%S')

#Checking loaded packages
print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),' Python Version', platform.python_version())
print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),' RDKit Version:', Chem.rdBase.rdkitVersion)

#Start time
start = time.time()

#File parameters
Infile = 'Sorted_23.sdf'
Outfile = 'test.sdf'

#Change this to get the number of desired molecules
how_many_to_pick = 126

mols =[]

#Read input file
suppl = Chem.SDMolSupplier(Infile)

#Generate fingerprints for valid molecules 
for mol in suppl:
    if mol is None: continue
    mols.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,2))

print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),' Total Valid Molecules', len(mols))

#Start picking diversed compounds
mmp = SimDivFilters.MaxMinPicker()
picks = mmp.LazyBitVectorPick(mols, len(mols),(how_many_to_pick+1))

#Convert to list
pickIndices = list(picks)

#Select the subset
subset = [suppl[mol] for mol in pickIndices]

print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),' Picking completed, Writing file initiated!')

writer = Chem.SDWriter(Outfile)

for mol in subset:
	if mol is None: continue
	writer.write(mol)

writer.close()

end = time.time()

print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),' Total time elaped %.2f sec'%(end - start))
