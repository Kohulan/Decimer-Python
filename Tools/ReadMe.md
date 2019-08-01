# Python Tools: For data curation, manipulation, etc..
- All codes are simply to implement please look at the requirements before using them. for further usage of the mentioned libraries please look at their main documentation.

## Required python libraries
 - [LZ4](https://lz4.github.io/lz4/)
 ```
 $ pip install lz4
 ```
 
 - pickle
 - datetime (optional)
 - itertools
 - numpy
 - multiprocessing
 - [Snappy](https://github.com/google/snappy) (if you want to try out Google's snappy algorithm)
 ```
 $ pip install python-snappy
 ```
 - [RDKIT](https://www.rdkit.org)
  - Installation instructions, for more details check the [main documentation.](https://www.rdkit.org/docs/Install.html)
    - Make sure you have [anaconda](https://www.anaconda.com) installed. Check [here](https://www.anaconda.com/distribution/ how to install.
  ```
  $ conda create -c rdkit -n my-rdkit-env rdkit
  $ conda activate my-rdkit-env
  ```
  - [Deepsmiles](https://chemrxiv.org/articles/DeepSMILES_An_Adaptation_of_SMILES_for_Use_in_Machine-Learning_of_Chemical_Structures/7097960/1)
  ```
  $ pip install --upgrade deepsmiles
  ```
  - [matplotlib](https://matplotlib.org)
  ```
  $ pip install -U matplotlib
  ```
  
  
### Data Reader
- Array_Reader.py
  - Reads text file containing 2D arrays and converts them into numpy arrays for further usage.

- ArrayReadParallel.py
  - Reads text file containing 2D arrays in compressed format into RAM, uncompress them over the fly, converts them into numpy arrays and uses uncompressed array further.

- ArrayReadInParallel_Batch
  - Does the same thing as ArrayReadParallel but can now be used for "Batching" of the data.

### Data compressors
- These were written to compress text data: especially the generated 2D text arrays.
- Snappy_compression.py
  -  Implementation of Google's [Snappy](https://en.wikipedia.org/wiki/Snappy_(compression)) algorithm. more details about Snappy in python can be found on [Google's github](https://github.com/google/snappy) page.

- Array_compressor.py
  - Implementation of [LZ4 algorithm](https://en.wikipedia.org/wiki/LZ4_(compression_algorithm)) on python to compress 2D arrays. Read the full [documentation.](https://lz4.github.io/lz4/)

- Compressed_DictReader.py
  - Can be used to decompress text data as well as data which was stored in a dictionary format inside a text file.

### Depictor & Splitter
- Depict_from_array.py
  -  Can be used to depict back greyscale images from 2D arrays using matplot libraries.

- Split_Smiles.py
  - Used to Split [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) into individual characters
  
### Deepsmiles encoder and decoder
- Deepsmiles_Encoder.py
  - Encodes canonical SMILES into Deepsmiles

- Deepsmiles_Decoder.py
  - Decodes Deepsmiles back to canonical SMILES

### RDKIT implementations
- SubsetPicker.py
  - RDKit implementation of [MaxMin algorithm](http://rdkit.blogspot.com/2017/11/revisting-maxminpicker.html) to pick maximum diverse subset from a large dataset.

- PariwiseTanimoto_Rdkit.py
  - Used to calculate the pairwise [Tanimoto](https://en.wikipedia.org/wiki/Jaccard_index) for given SMILES list in a text file.

- parallel_PairwiseTanimoto_Rdkit.py
  - Implementation of pairwise Tanimoto and uses multi-threading to improve the speed and reduce the time in calculating pairwise Tanimoto.

- Tanimoto_Calculator_for_predictions.py
  - Tanimoto calculations implementation for predictions
  
## References
1- https://lz4.github.io/lz4/
2- https://python-lz4.readthedocs.io/en/latest/
3- https://en.wikipedia.org/wiki/LZ4_(compression_algorithm)
4- https://en.wikipedia.org/wiki/Snappy_(compression)
5- https://github.com/google/snappy
6- SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules
David Weininger
Journal of Chemical Information and Computer Sciences 1988 28 (1), 31-36
DOI: 10.1021/ci00057a005
7- RDKit: Open-source cheminformatics; http://www.rdkit.org
8- https://www.slideshare.net/NextMoveSoftware/recent-improvements-to-the-rdkit
9- [DeepSMILES:](https://github.com/nextmovesoftware/deepsmiles) An Adaptation of SMILES for Use in Machine-Learning of Chemical Structures
Noel O'Boyle Andrew Dalke
DOI: https://doi.org/10.26434/chemrxiv.7097960.v1

