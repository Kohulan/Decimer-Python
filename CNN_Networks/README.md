# Convolutional Neural Networks to work on Compressed data of 2D Image arrays

## Contents :
```
...
├── CNN_2layer20190922.py                     #The initial working script, written to deploy a CNN on our current compressed dataset.
├── CNN_Multiclass_New_Deep_Architecture.py   #CNN for multiclass classification, with more convolutional layers and pooling layers which exhibits the "Deep" architecture.
├── Cnn_alexnet.py                            #CNN implementation of the Famous [Alexnet](https://en.wikipedia.org/wiki/AlexNet) Architecture by Geoffrey Hinton's [paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
├── Cnn_minimal.py                            #CNN network with 2 convolutional layers
├── Cnn_modified_alexnet.py                   #Modified version of Alexnet architecture with better report writing
└── Deep_CNN_Alexnet_implementation.py        #Reimplementation of Alexnet with better Test batch usage
```

## Requirements :
- Networks can run on CPU or GPUs, GPUs are much faster.
- All codes were written in python 2.7and can be used in Python 3.6 (Modification ongoing). Tensorflow libraries v1.13 was used in here.
- Install all the required libraries, for more details please look at the [Python_Requirements.txt](https://github.com/Kohulan/Decimer-Python/blob/master/Python_Requirements.txt).

### To run on GPUs
  - Please do install nVidia GPU drivers, CUDA drivers and cuDNNs. Please refer [here](https://github.com/Kohulan/CUDA-10-with-Tensoflow2.0-Installation-Guide) , for how to install.
  - Please install [Tensorflow-GPU](https://www.tensorflow.org/install/gpu).

## Usage:
- A user must have all the required libraries installed, Data should be packaged as compressed text files.
    - Links for the produced data will be updated in the future.
- Please do have the necessary storage space available and check the available RAM size on the PC.
- Scripts can be downloaded directly from the repository.

## License:
- This project is licensed under the MIT License - see the [LICENSE](https://github.com/Kohulan/Decimer-Python/blob/master/LICENSE) file for details

## Author:
- [Kohulan](github.com/Kohulan)
