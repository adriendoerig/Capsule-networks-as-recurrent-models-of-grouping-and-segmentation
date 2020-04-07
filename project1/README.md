# Experiment 1: Crowding and Uncrowding Naturally Occur in CapsNets
This code can be used to reproduce the results shown in 'Capsule Networks but not Classic CNNs Explain
Global Visual Processing' by Doerig, Schmittwilken, Sayim, Manassi, and Herzog (2020)

Link to preprint: [Link](https://www.researchgate.net/publication/335472170_Capsule_Networks_as_Recurrent_Models_of_Grouping_and_Segmentation
)

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

## parameters.py
This file contains most of the parameters used for running the code. It is recommended to get familiar and check the parameters (such as the data paths, the stimulus parameters, and the network parameters) before executing the code.

## Running the code
If you are running the code for the first time, create the datasets used for training and testing by running
```
python make_tfrecords.py
```
This will create a ./data folder with a variety of .tfrecords files containing all the input stimuli for the network during training, validation and testing.

Next, run
```
python capser_main.py
```
This will train and test the selected number of networks (default: 10).
By default, this will also automatically initiate the get_reconstructions.py code to create plots containing exemplary reconstructions of the input images during testing as included in the paper.

In order to create the final performance plots averaged over all networks, run
```
python plot_final_performance.py
```

## Acknowledgements
We would like to thank Lynn Schmittwilken ([Link](l.schmittwilken@tu-berlin.de)) for the creation and execution of the code.