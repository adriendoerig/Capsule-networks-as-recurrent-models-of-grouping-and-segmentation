# Experiment 1: Crowding and Uncrowding Naturally Occur in CapsNets
This code can be used to reproduce the results shown in 'Capsule Networks but not Classic CNNs Explain Global Visual Processing' by Doerig, Schmittwilken, Sayim, Manassi, and Herzog (2020)

Link to preprint: [Preprint Capsule Networks but not Classic CNNs Explain Global Visual Processing](https://www.researchgate.net/publication/335472170_Capsule_Networks_as_Recurrent_Models_of_Grouping_and_Segmentation)

## Prerequisites
This code was tested with the following modules/packages. It is not ensured that it will run with any modules/packages that are not listed here.
* Python 3.5.5
* Python 3.6
* TensorFlow 1.10
* TensorFlow 1.12
* TensorFlow-gpu 1.10
* Numpy 1.14.3
* Numpy 1.15.4
* Skimage 0.13.1
* Skimage 0.14.0

## *parameters.py*
This file contains most of the parameters used for running the code. It is recommended to get familiar and check the parameters (such as the data paths, the stimulus parameters, and the network parameters) before executing the code.

## Running the code
If you are running the code for the first time, create the datasets used for training and testing by running
```
python make_tfrecords.py
```
This will create a *./data* folder with a variety of *.tfrecords* files containing all the input stimuli for the network during training, validation and testing.

Next, run
```
python capser_main.py
```
This will train and test the selected number of networks (default: 10). By default, all output files will be saved in *./data/_logs_1*. After executing the code, this logdir contains the final results (means and errors) averaged over all trained networks, a *.txt* copy of the *parameters.py*, as well as the individual network outputs in the folders *./data/_logs_1/0* to *./data/_logs_1/9*.

By default, running *capser_main.py* will automatically initiate *get_reconstructions.py* to create plots containing exemplary reconstructions of the input images during testing as included in the paper. If this is not the case, you can create figures with the reconstructed input images during training by running
```
python get_reconstructions.py
```
The reconstructions will be saved in *./data/_logs_1/network_ID/reconstructions* (e.g. *./data/_logs_1/0/reconstructions*).

In order to create the final performance plot (*./data/_logs_1/final_performance.png*) averaged over all networks, run
```
python plot_final_performance.py
```

## Acknowledgements
We would like to thank Lynn Schmittwilken (l.schmittwilken@tu-berlin.de) for the creation and execution of the code.