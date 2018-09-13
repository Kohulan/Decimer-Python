'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by : Kohulan.R on 2018 07 26
'''
#Snappy compression and decompression

import snappy
#Compression
def compress_snappy(path):
	directory_path = path+'.snappy'
	with open(path, 'rb') as in_file:
		with open(path_to_store, 'w') as out_file:
			snappy.stream_compress(in_file, out_file)
			out_file.close()
			in_file.close()
	return directory_path

#Decompression
def decompress_snappy(path):
	directory_path = path[:-7]
	with open(path, 'rb') as in_file:
		with open(path_to_store, 'w') as out_file:
			snappy.stream_decompress(in_file, out_file)
			out_file.close()
			in_file.close()
	return directory_path

decompress_snappy('file.name.snappy')
#compress_snappy('Labels_train.csv')
