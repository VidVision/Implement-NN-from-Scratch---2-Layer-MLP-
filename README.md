# Implement-NN-from-Scratch---2-Layer-MLP-
In this work two networks have been implemented from scratch: a simple softmax regression and a two-layer multi-layer perceptron (MLP).

A simple pipeline of training neural networks is implemented to recognize MNIST Handwritten Digits (http://yann.lecun.com/exdb/mnist/).
The code includes implementation of two neural network architectures along with the code to load data, train and optimize these networks.

- **main.py**
  contains the major logic of this project. It can be executed by invoking the following command where the yaml file contains all the
hyper-parameters.
$ python main.py --config configs/<name_of_config_file>.yaml

- **./configs**
Contains the default haper parameter files and *config_exp.yaml* which is used for HP tuning.
The script trains a model with the number of epochs specified in the config file. At the end of each epoch, the script evaluates the model on the
validation set. After the training completes, the script finally evaluates the best model on the test data.

## Python and dependencies
Python 3 is used in this project.

- **environment.yaml**
Contains a list of needed libraries. 
$ conda env create -f environment.yaml

## Data Loading ##
- **./data**
Download the MNIST dataset with  provided script under ./data by:
$ cd data
$ sh get_data . sh
$ cd . . /
Microsoft Windows 10 Only
C: \ c o de  folder > cd data
C: \ c o de  folder \ data> get_data . bat
C: \ c o de  folder \ data> cd . .
The dataset is already downloaded so this step can be skipped.

### Data Preparation ##
- **./utils.py**
To avoid the choice of hyper-parameters overfits the training data, it is a common practice to split the training dataset into the actual training data and validation data and perform hyper-parameter tuning based on results on validation data. Additionally, in deep learning, training data is often forwarded to models in batches for faster training time and noise reduction.
In our pipeline, we first load the entire MNIST data into the system, followed by a training/validation split on the training set. We simply use the first
80% of the training set as our training data and use the rest training set as our validation data. We also want to organize our data (training,
validation, and test) in batches and use different combination of batches in different epochs for training data.
These are performed in
 * load_mnist_trainval in **./utils.py** for training/validation split
 * generate_batched_data in **./utils.py** to organize data in batches

## Model Implementation ##
- **./models**
two networks are implemented from scratch: 
* a simple softmax regression
* a two-layer multi-layer perceptron (MLP). 
Definitions of these classes can be found in ./models.

The Softmax Regression is composed by a fully-connected layer followed by a ReLU activation. The two-layer MLP is composed by two fully-connected layers with a Sigmoid Activation in between.

## Optimizer ##
- **./optimizer**
* _base_optimizer.py*
* sgd.py*
An optimizer is used to update weights of models. The optimizer is initialized with a specific learning rate and a regularization coefficients. Before
updating model weights, the optimizer applies L2 regularization on the model (Cross Entropy Loss + L2)

