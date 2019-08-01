# Neural Networks to work on Compressed data of 2D Image arrays

## Contents :
```
...
├── 3Layer_Network_for_Parameter_Optimization.py                     #3 Layered Perceptron used for parameter optimization.
├── 3layer_net_test.py                                               #Initial commit of a working 3 Layered Perceptron.
├── 3layer_net_test_compressed.py                                    #3 Layered Perceptron which reads compressed files and uncompress them while training.
├── 3layer_net_test_paralleldata_processing.py                       #Initial 3 Layered Perceptron which uses parallel dataprocessing, implementation of python multiprocessing.
├── 4_Layer_MLP.py                                                   #4 Layered network fully functional
├── 4_layer_net_Parameter_optimization.py                            #4 Layered Network used for parameter optimizing
└── tensorboard_3layer_net                                           #Tensorboard implementation inside a 3 Layered Network
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
