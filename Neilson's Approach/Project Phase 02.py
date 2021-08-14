#!/usr/bin/env python
# coding: utf-8

# # <font color=brown>Marsland's approach on MNIST</font>
# **Date:** Tuesday, April 13, 2021

# ## Phase 2: Experiments with Nielsen’s approach on MNIST
# 1.	Use Michael Nielsen’s network2.py  to train on the first 1000 samples for 100 epochs using his default values for other parameters.  Plot the evaluation accuracy and training accuracy on one plot, and the evaluation cost and training cost on another plot.
# 2.	Test on learning rates 0.25, 0.5, 0.7 using 3 different trials for each learning rate.  Observe and plot the performance in terms of evaluation accuracy for each learning rate.
# 3.	Test on mini-batches of size 5 and 20 using 3 different trials for each size and Nielsen’s default values for other parameters.  Observe adopt the performance in terms of evaluation accuracy for each mini-batch size.
# 4.	Train on the full 50k images for 30 epochs using the same values as Nielsen for other parameters except changing L2 parameter to 5 because of much larger training set and using the test set as the evaluation set.  Plot the evaluation accuracy and training accuracy.  Document the performance of the net.
# 5.	Submit your Jupyter notebook that includes the code, outputs, descriptive comments, observations and conclusions
# 

# ## <font color=sienna>Introduction</font>
# 
# The MNIST or Mixed National Institute of Standards and Technology dataset is a collection of handwritten digits. The data set contains 60,000 training images and 10,000 testing images of handwriting samples. All the images are in grayscale and are 28 by 28 pixels in size. The MNIST data set is widely used in machine learning for image classification. In this project we look at two different approaches and compare them. The first is Marsland’s approach that uses MLP on MNIST and trains only 200 examples. The second approach is Nielsen’s. Nielsen’s approach trains on 1000 training examples and uses 30 hidden layer nodes, Cross-entropy, and L2 regularization. For this project however, we slightly modified the two approaches for experimental purposes. 
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/220px-MnistExamples.png" width="300" align="center"/><center> **Figure 1.1:** Sample images from MNIST test dataset</center>

# ## <font color=sienna>Experiments</font>

# The copy of Michael Nielsen’s network2.py 

# In[1]:


"""network2.py
~~~~~~~~~~~~~~
An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.
"""

#### Libraries
# Standard library
import json
import random
import sys
import shelve

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

data_file_name = 'exp_data'

#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data)
            print
        return evaluation_cost, evaluation_accuracy,             training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


# In[2]:


"""Loading Data"""
import mnist_loader # Loading mnist dataset. Some minor changes made to the file
import pylab as pl

training_data, validation_data, test_data = mnist_loader.load_data_wrapper() #loading train, val, test data.
training_data = list(training_data)

"""Define net with Cross entropy and 3 layers"""
net = Network([784, 30 , 10],cost=CrossEntropyCost)


# In[4]:


data_file_name = 'exp_data' # Not required if network2.py was executed first


# ### 1.1. Train on the first 1000 samples for 100 epochs using his default values for other parameters.

# In[3]:


"""Experiment: Part 1"""
evaluation_cost, evaluation_accuracy,training_cost, training_accuracy = net.SGD(training_data[:1000], 100, 10, 0.5, 
        lmbda = 0.1,evaluation_data=validation_data,monitor_evaluation_accuracy=True,monitor_evaluation_cost=True,
        monitor_training_accuracy=True,monitor_training_cost=True)

print("(Note: Experiment 1 Performed!)")

"""Test Result Accumulation"""
data = shelve.open(data_file_name)
try:
    data['exp_1_epoches'] = range(0,100)
    data['exp_1_evaluation_cost'] = [x for x in evaluation_cost]
    data['exp_1_training_cost'] = [x for x in training_cost]
    data['exp_1_evaluation_accuracy'] = [x/10000.0 for x in evaluation_accuracy]
    data['exp_1_training_accuracy'] = [x/1000. for x in training_accuracy]
finally:
    data.close()


# ### 1.2. Plot the evaluation accuracy and training accuracy on one plot, and the evaluation cost and training cost on another plot

# In[5]:


"""Results From Part 1 Experiments"""

data = shelve.open(data_file_name)
try:
    figureName = "Figure 3.1: Cost vs Epoch Curve for Test 1 over 100 Epoches."
    plt.figure(figsize=(8,5))
    plt.plot(data['exp_1_epoches'], data['exp_1_evaluation_cost'], label='Evaluation')
    plt.plot(data['exp_1_epoches'], data['exp_1_training_cost'], label='Training')
    plt.title(figureName)
    plt.xlabel('Epoches')
    plt.ylabel('Cost')
    plt.legend(loc='best')
    plt.show()

    figureName = "Figure 3.2: Accuracy vs Epoch Curve for Test 1 over 100 Epoches."
    plt.figure(figsize=(8,5))
    plt.plot(data['exp_1_epoches'], data['exp_1_evaluation_accuracy'], label='Evaluation')
    plt.plot(data['exp_1_epoches'], data['exp_1_training_accuracy'], label='Training')
    plt.title(figureName)
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
finally:
    data.close()


# Min Evaluation Cost Epoch 10 : 18.79% 
# Cost on training data: 0.187957671557
# Accuracy on training data: 995 / 1000
# Cost on evaluation data: 0.699844652398
# Accuracy on evaluation data: 8901 / 10000
# 
# Min Training Cost Epoch 99 : 8.98%
# Cost on training data: 0.0898346736957
# Accuracy on training data: 1000 / 1000
# Cost on evaluation data: 0.787557422199
# Accuracy on evaluation data: 8883 / 10000
# 
# Max Evaluation Accuracy Epoch 19 : 89.14% -> 0.89
# Cost on training data: 0.118116794792
# Accuracy on training data: 999 / 1000
# Cost on evaluation data: 0.721696647455
# Accuracy on evaluation data: 8914 / 10000
# 
# Max Training Accuracy Epoch 20 : 100% -> 1
# Cost on training data: 0.115839324687
# Accuracy on training data: 1000 / 1000
# Cost on evaluation data: 0.728386684838
# Accuracy on evaluation data: 8886 / 10000

# ### 2.1. Test on learning rates 0.25, 0.5, 0.7 using 3 different trials for each learning rate.

# In[6]:


"""Experiment: Part 2"""
epoch = 100
learning_rates = [0.25,0.5,0.7] 
exp_2_res_list=[]
for lr in learning_rates : 
    i=0
    print "---------Learning Rate :",lr,"------------"
    while i<3 :
        print "----Loop ",i+1,"---------"
        evaluation_cost, evaluation_accuracy,training_cost, training_accuracy = net.SGD(training_data[:1000], epoch, 
                    10, lr, lmbda = 0.1,evaluation_data=validation_data,monitor_evaluation_accuracy=True,
                    monitor_evaluation_cost=True,monitor_training_accuracy=True,monitor_training_cost=True)

        exp_2_res_list.append([lr,i+1,evaluation_accuracy])
        i = i+1

print("(Note: Experiment 2 Performed!)")

"""Result Accumulation"""
data = shelve.open(data_file_name)
try:
    data['exp_2_epoches'] = range(0,100)
    data['exp_2_test_1_learning_rates'] = [i[0] for i in exp_2_res_list if i[1]==1]
    data['exp_2_test_2_learning_rates'] = [i[0] for i in exp_2_res_list if i[1]==2]
    data['exp_2_test_3_learning_rates'] = [i[0] for i in exp_2_res_list if i[1]==3]
    data['exp_2_test_1_eval_accuracy'] = [i[2][-1]/10000.0 for i in exp_2_res_list if i[1]==1]
    data['exp_2_test_2_eval_accuracy'] = [i[2][-1]/10000.0 for i in exp_2_res_list if i[1]==2]
    data['exp_2_test_3_eval_accuracy'] = [i[2][-1]/10000.0 for i in exp_2_res_list if i[1]==3]
    data['exp_2_res_list'] = exp_2_res_list
finally:
    data.close()


# ### 2.2. Observe and plot the performance in terms of evaluation accuracy for each learning rate.

# In[7]:


"""Results From Part 2 Experiments"""
data = shelve.open(data_file_name)
try:
    print 
    print "Test 1 : learning rates",data['exp_2_test_1_learning_rates'],", Evaluation Accuracy",data['exp_2_test_1_eval_accuracy']
    print "Test 2 : learning rates",data['exp_2_test_2_learning_rates'],", Evaluation Accuracy", data['exp_2_test_2_eval_accuracy']
    print "Test 3 : learning rates",data['exp_2_test_3_learning_rates'],", Evaluation Accuracy", data['exp_2_test_3_eval_accuracy']
    figureName = "Figure 3.3: Learning Rates vs 100th epoch Evaluation accuracy"
    plt.figure(figsize=(8,5))
    plt.scatter(data['exp_2_test_1_learning_rates'],  data['exp_2_test_1_eval_accuracy'],color='red' ,marker = '^')
    plt.plot(data['exp_2_test_1_learning_rates'], data['exp_2_test_1_eval_accuracy'],color='red',label=str('Test 1'))
   
    plt.scatter(data['exp_2_test_2_learning_rates'], data['exp_2_test_2_eval_accuracy'],color='blue' ,marker = 'x')
    plt.plot(data['exp_2_test_2_learning_rates'], data['exp_2_test_2_eval_accuracy'],color='blue',label=str('Test 2'))
   
    plt.scatter(data['exp_2_test_3_learning_rates'], data['exp_2_test_3_eval_accuracy'],color='green',marker = 'o')
    plt.plot(data['exp_2_test_3_learning_rates'], data['exp_2_test_3_eval_accuracy'],color='green',label=str('Test 3'))
   
    axes = plt.gca()
    #axes.set_xlim([xmin,xmax])
    axes.set_ylim([0.88,0.90])
    plt.title(figureName)
    plt.xlabel('Learning Rate')
    plt.ylabel('Evaluation Accuracy')
    plt.legend(loc='best')
    plt.show()
finally:
    data.close()


# #### Max Evaluation Accuracy 
# 
# The accuracy varied among individual epoches but was almost consistent throughout ranging between 88.95%-89.10%. The best accuracy was for learning rate 0.7 with 89.10%. There wasn't a significant difference with the change in learning rate.
# 
# For learning rate 0.25
# 
# Test 1 : Epoch 79 Cost on training data: 0.088407459694 Accuracy on training data: 1000 / 1000 Cost on evaluation data: 0.78736028613 Accuracy on evaluation data: 8897 / 10000
# 
# Test 2 : Epoch 67 Cost on training data: 0.0872564386367 Accuracy on training data: 1000 / 1000 Cost on evaluation data: 0.786081734897 Accuracy on evaluation data: 8895 / 10000
# 
# Test 3 : Epoch 98 Cost on training data: 0.0861482812036 Accuracy on training data: 1000 / 1000 Cost on evaluation data: 0.786817066286 Accuracy on evaluation data: 8897 / 10000
# 
# For learning rate 0.5
# 
# Test 1 : Epoch 56 Cost on training data: 0.0854119707339 Accuracy on training data: 1000 / 1000 Cost on evaluation data: 0.784504655229 Accuracy on evaluation data: 8910 / 10000
# 
# Test 2 : Epoch 46 Cost on training data: 0.0846435121285 Accuracy on training data: 1000 / 1000 Cost on evaluation data: 0.784699264208 Accuracy on evaluation data: 8908 / 10000
# 
# Test 3 : Epoch 40 Cost on training data: 0.0836692265152 Accuracy on training data: 1000 / 1000 Cost on evaluation data: 0.777918313703 Accuracy on evaluation data: 8907 / 10000
# 
# For learning rate 0.7
# 
# Test 1 : Epoch 48 Cost on training data: 0.0830364517493 Accuracy on training data: 1000 / 1000 Cost on evaluation data: 0.776500117537 Accuracy on evaluation data: 8908 / 10000
# 
# Test 2 : Epoch 28 Cost on training data: 0.0827193434588 Accuracy on training data: 1000 / 1000 Cost on evaluation data: 0.775547951186 Accuracy on evaluation data: 8910 / 10000
# 
# Test 3 : Epoch 23 Cost on training data: 0.0819855759689 Accuracy on training data: 1000 / 1000 Cost on evaluation data: 0.783979675921 Accuracy on evaluation data: 8909 / 10000 

# ### 3.1. Test on mini-batches of size 5 and 20 using 3 different trials for each size and Nielsen’s default values for other parameters.

# In[8]:


"""Experiment: Part 3"""
epoch = 100
minibatch = [5,20] 
exp_3_res_list=[]
for mb in minibatch : 
    i=0
    print "---------Mini Batch : ",mb,"------------"
    while i<3 :
        print "----Loop ",i+1,"---------"
        evaluation_cost, evaluation_accuracy,training_cost, training_accuracy = net.SGD(
            training_data[:1000], epoch, mb, 0.5, lmbda = 0.1,evaluation_data=validation_data,monitor_evaluation_accuracy=
            True, monitor_evaluation_cost=True, monitor_training_accuracy=True,monitor_training_cost=True)

        exp_3_res_list.append([mb,i+1,evaluation_accuracy])
        i = i+1

print("(Note: Experiment 3 Performed!)")

"""Result Accumulation"""
data = shelve.open(data_file_name)
try:
    data['exp_3_epoches'] = range(0,100)
    data['exp_3_test_1_mini_batch_sizes'] = [i[0] for i in exp_3_res_list if i[1]==1]
    data['exp_3_test_2_mini_batch_sizes'] = [i[0] for i in exp_3_res_list if i[1]==2]
    data['exp_3_test_3_mini_batch_sizes'] = [i[0] for i in exp_3_res_list if i[1]==3]
    data['exp_3_test_1_eval_accuracy'] = [i[2][-1]/10000.0 for i in exp_3_res_list if i[1]==1]
    data['exp_3_test_2_eval_accuracy'] = [i[2][-1]/10000.0 for i in exp_3_res_list if i[1]==2]
    data['exp_3_test_3_eval_accuracy'] = [i[2][-1]/10000.0 for i in exp_3_res_list if i[1]==3]
    data['exp_3_res_list'] = exp_3_res_list
finally:
    data.close()


# ### 3.2. Observe adopt the performance in terms of evaluation accuracy for each mini-batch size.

# In[12]:


"""Results From Part 3 Experiments"""
data = shelve.open(data_file_name)
try:
    print 
    print "Test 1 : Mini Batch",data['exp_3_test_1_mini_batch_sizes'],", Evaluation Accuracy",data['exp_3_test_1_eval_accuracy']
    print "Test 2 : Mini Batch",data['exp_3_test_2_mini_batch_sizes'],", Evaluation Accuracy", data['exp_3_test_2_eval_accuracy']
    print "Test 3 : Mini Batch",data['exp_3_test_3_mini_batch_sizes'],", Evaluation Accuracy", data['exp_3_test_3_eval_accuracy']
    
    figureName = "Figure 3.4: Evaluation Accuracy vs Mini Batch Size"
    plt.figure(figsize=(8,5))
    plt.scatter(data['exp_3_test_1_mini_batch_sizes'], data['exp_3_test_1_eval_accuracy'],color='red' ,marker = '^')
    plt.plot(data['exp_3_test_1_mini_batch_sizes'], data['exp_3_test_1_eval_accuracy'],color='red',label=str('Test 1'))
    
    plt.scatter(data['exp_3_test_2_mini_batch_sizes'], data['exp_3_test_2_eval_accuracy'],color='blue' ,marker = 'x')
    plt.plot(data['exp_3_test_2_mini_batch_sizes'], data['exp_3_test_2_eval_accuracy'],color='blue',label=str('Test 2'))
    
    plt.scatter(data['exp_3_test_3_mini_batch_sizes'], data['exp_3_test_3_eval_accuracy'],color='green',marker = 'o')
    plt.plot(data['exp_3_test_3_mini_batch_sizes'], data['exp_3_test_3_eval_accuracy'],color='green',label=str('Test 3'))

    axes = plt.gca()
    axes.set_xlim([4,22])
    axes.set_ylim([0.882,0.89])
    plt.title(figureName)
    plt.xlabel('Mini Batch Size')
    plt.ylabel('Evaluation Accuracy')
    plt.legend(loc='best')
    plt.show()
finally:
    data.close()


# #### Max Evaluation Accuracy
# 
# Evaluation Accuracy increases with the increase in mini batch size for the last epoch shown above in the plot. Looking at the values of all the epoches from above, we can see that with the increase in mini batch size, the accuracy converges and becomes more consistent. Here in the plot, when mini batch size is 5 the accuracy ranges from 88.78%-88.88% but for mini batch size 20, it ranges from 88.87%-88.91%.

# ### 4.1. Train on the full 50k images for 30 epochs using the same values as Nielsen for other parameters except changing L2 parameter to 5 because of much larger training set. 

# In[13]:


"""Experiment: Part 4"""
evaluation_cost, evaluation_accuracy,training_cost, training_accuracy = net.SGD(training_data, 30, 10, 0.5, lmbda = 5.0,
        evaluation_data=validation_data,monitor_evaluation_accuracy=True,monitor_evaluation_cost=True,
        monitor_training_accuracy=True,monitor_training_cost=True)

print("(Note: Experiment 4 Performed!)")

"""Result Accumulation"""
data = shelve.open(data_file_name)
try:
    data['exp_4_epoches'] = range(0,30)
    data['exp_4_evaluation_accuracy'] = [x/10000.0 for x in evaluation_accuracy]
    data['exp_4_training_accuracy'] = [x/50000.0 for x in training_accuracy]
finally:
    data.close()


# ### 4.2. Plot the evaluation accuracy and training accuracy. 

# In[14]:


"""Results From Part 4 Experiments"""
data = shelve.open(data_file_name)
try:
    figureName = "Figure 3.5: Accuracy vs Epoch Curve Obtained from Test 4 on 30 Epoches and Full Data Set"
    plt.figure(figsize=(8,5))
    plt.plot(data['exp_4_epoches'], data['exp_4_evaluation_accuracy'], label='Evaluation')
    plt.plot(data['exp_4_epoches'], data['exp_4_training_accuracy'], label='Training')
    plt.title(figureName)
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
finally:
    data.close()


# Max Accuracy is for training data set in comparison to evaluation data set.

# ## <font color=sienna>Conclusion</font>

# Both approaches have merit and provide insight into training the nets for the dataset. The advantage of the Marsland approach is that there is a larger focus on number of nodes than any other parameter which would help optimize the net for number of nodes quite well. The disadvantage is that multiple epochs are not used and learning rate is not as important as number of nodes. The advantage to using Nielsen’s approach is that the focus lies more on learning rates, batch size, and epochs with the highest emphasis being placed on learning rate and epochs. If both were to be used in tandem the focus on number of nodes could be used from the Marsland approach with a learning rate of 0.5 which was determined to be the best from the Nielsen approach. After the number of nodes was adjusted for the 0.5 learning rate the epoch and mini batch parameters could be retested and the best results could be used from the combination of the two approaches. This setup would result in the most accurate classification process and produce less error while also decreasing the number of iterations necessary to get learning rates or epochs from using either approach alone. 

# In[ ]:




