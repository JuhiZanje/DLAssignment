# **Fashion MNIST classification using a "numpy" Feed forward Neural Network**

This folder contains the code base for Assignment 1 as part of CS6910: Deep Learning Fundamentals.

### Introduction:
The problem statement involves building and training a 'plain vanilla' Feed Forward Neural Network from scratch using primarily Numpy package in Python.  

The code base now has the following features:
1. Forward and backward propagation are coded using Matrix operations.
2. A MultiLayerPerceptron class to instantiate the neural network object for specified set of hyperparameters, namely the number of layers, hidden neurons, activation function, optimizer, weight decay,etc.This class contains all the optimizers as well.
3. For Loss and Activation functions I have made a common class.
4. A colab notebook containing the entire code to train and validate the model from scratch.

## Dataset

Fashion MNIST data set has been used in this assignment.
size of data:
Train_data  - 60000
Test_data - 20000
Validation_data - 6000

During the hyperparameter optimization stage, 10% of the training dataset, consisting of around 6000 images and their corresponding labels, is set aside for validation for each hyperparameter configuration. The remaining 90%, approximately 54000 images, are used for training the model.
After identifying the best configuration using tools like Weights & Biases (wandb) with either Random Search or Bayesian Optimization, the full training dataset is utilized to train the selected best model configuration. Subsequently, the test accuracy is computed, followed by the generation of a confusion matrix for evaluation purposes.

## GitHub Files to use
**train.py->**
* Project Name (wandb_project) default set to 'CS6910 - Assignment 1'
* Project name is space saperated thus when using it for commands you need to use it in **inverted comas (ie. in string format)**
* when using Project name in command use as : **"CS6910 - Assignment 1"** (with inverted commas)
* Or use the default setting
* Entity Name (wandb_entity) in project default set to 'Juhi_Zanje'
* as given in the question description.

**Assignment_1_jupitor->**
* This is a Jupitor file which contains all the Questions given in the assignment section wise.
* To run this File you will have to run file from starting section , one by one.
* When needed Will have to install some packages.
* By running this file you can get all the outputs I have shown in my Report.


### String Inputs for Function Selection
You can select various options for different functions using specific strings:
- **Loss Function**: 'mean_squared_error' or 'cross_entropy'
- **Optimization Function**: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', or 'nadam'
- **Activation Function**: 'sigmoid', 'tanh', 'ReLU' ,'identity'
- **Weight Initialization**: 'random' or 'Xavier'


## Results:
For the plain vanilla feed forward neural network implemented, the maximum test accuracy reported was 88.8% on the Fashion MNIST dataset and ~97.81% on the MNIST hand written datasets.
One of the model configuration chosen to be  the best is as follows:

- Number of Hidden Layers - 4
- Number of Hidden Neurons - 256
- L2 Regularisation - 0.0005
- Activation - ReLu
- Initialisation - Xavier
- Optimiser - nadam
- Learning Rate - 0.001
- Batch size - 32