# Experiment 2: The role of recurrent processing
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
This will train and test the selected number of networks (default: 50). By default, all output files will be saved in *./data/_logs_1*. After executing the code, this logdir contains the final results (means and errors) averaged over all trained networks, a *.txt* copy of the *parameters.py*, as well as the individual network outputs in the folders *./data/_logs_1/0* to *./data/_logs_1/9* that each contain network outputs and individual performances for each routing iteration (e.g. *./data/_logs_1/0/iter_routing_1*).

In order to create the final performance plots for the model averaged over all networks (*./data/_logs_1/final_performance_model.png*) as well as the performance plots based on the human experiment (*./data/_logs_1/final_performance_humans.png*), run
```
python plot_final_performance.py
```
The code will also perform a statistical analysis comparing the distribution of performances over time/routing iterations (more precisely: the performance slope) between the different configurations (lines vs. cuboids).

## Acknowledgements
We would like to thank Lynn Schmittwilken (l.schmittwilken@tu-berlin.de) for the creation and execution of the code.