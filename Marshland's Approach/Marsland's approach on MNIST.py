#!/usr/bin/env python
# coding: utf-8

# # <font color=brown>Marsland's approach on MNIST</font>
# **Date:** Friday, March 26, 2021

# ## Phase 1: Experiments with Marsland’s approach on MNIST
# Marsland’s implementation of MLP on MNIST trains only 200 examples, uses
# - Sum squared error
# - .9 momentum
# - Learning rate .1
# - Early stopping
# 
# Phase 1: Experiments with Marsland’s approach on MNIST
# 1.	use Marsland’s implementation to create nets with hidden layer nodes of 1, 2, 5, 10, 20, 30 and train them on the first 1000 examples using Marsland’s learning rate of 0.1.
# a.	Set the seed on the random number generator and demonstrate that the results are the same for each of 2 trials using the same seed.  
# b.	Use 3 different seeds to train from 3 different initial weights and record the best results for each architecture and note the number of hidden layer nodes in the best performing architecture. 
# 2.	Create a net with the number of hidden layer nodes in 1b above and train it on the 1000 examples.  Test learning rates of 0.05, 0.1, 0.3, 0.5, running with 3 different seeds for each learning rate.  Observe and document the best performance for each learning rate, and the best overall performance.
# 3.	Submit your Jupyter notebook that includes code, outputs, descriptive comments, observations, conclusions.

# ## <font color=sienna>Introduction</font>
# 
# Marsland’s implementation of MLP on MNIST trains only 200 examples. This phase 1 of the project will use his implementation of MLP on MNIST to train 1000 examples instead. Marsland’s MLP uses the “sum squared error”, .9 momentum, a learning rate of .1, and also early stopping. We will experiment with this implementation he has provided of MLP by ways of changing hidden layer nodes between 1/2/5/10/20/30 nodes, using 4 different seeds to train from 3 different weights, and noting the best performing hidden layer architecture. We will additionally be changing learning rates through 0.05, 0.1, 0.3, and 0.5 while running the 4 different seeds for each of those learning rates.

# ## <font color=sienna>Experiments</font>

# In[1]:


import pylab as pl
import numpy as np
import cPickle, gzip


# The copy of MLP class provided by author.

# In[2]:


class mlp:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,inputs,targets,nhidden,seed,beta=1,momentum=0.9,outtype='logistic'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden
        self.seed = seed

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        # Initialise network
        np.random.seed(seed)
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
    
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
           # print count
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            
        #print "Stopped", new_val_error,old_val_error1, old_val_error2
        return new_val_error
    	
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        change = range(self.ndata)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
            
        for n in range(niterations):
    
            self.outputs = self.mlpfwd(inputs)

            error = 0.5*np.sum((self.outputs-targets)**2)
           # if (np.mod(n,100)==0):
           #     print "Iteration: ",n, " Error: ",error    

            # Different types of output neurons
            if self.outtype == 'linear':
            	deltao = (self.outputs-targets)/self.ndata
            elif self.outtype == 'logistic':
            	deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata 
          #  else:
          #  	print "error"
            
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
                      
            updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2
                
            # Randomise order of inputs (not necessary for matrix-based calculation)
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]
            
    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print "error"

    def confmat(self,inputs,targets,seed,hidden_layer_nodes,learn_rate,test):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))
        filename = "Experiment NHNode-{0} LRate-{1} Seed-{2} Test-{3}.npy".format(hidden_layer_nodes,learn_rate,seed,test)
        print "Confusion matrix is:"
        print cm
        np.save(filename, cm)
        percentage_Correct = np.trace(cm)/np.sum(cm)*100
        print "Percentage Correct: ",percentage_Correct
        return percentage_Correct


# Some of the changes that we did apart from the author's code is that we added a new file creation logic where we save all the training and testing done in the process.

# In[3]:


# Read the dataset in (code from sheet)
f = gzip.open('mnist.pkl.gz','rb')
tset, vset, teset = cPickle.load(f)
f.close()

nread = 1000
# Just use the first few images
train_in = tset[0][:nread,:]

# This is a little bit of work -- 1 of N encoding
# Make sure you understand how it does it
train_tgt = np.zeros((nread,10))
for i in range(nread):
    train_tgt[i,tset[1][i]] = 1

test_in = teset[0][:nread,:]
test_tgt = np.zeros((nread,10))
for i in range(nread):
    test_tgt[i,teset[1][i]] = 1

# We will need the validation set
valid_in = vset[0][:nread,:]
valid_tgt = np.zeros((nread,10))
for i in range(nread):
    valid_tgt[i,vset[1][i]] = 1
    #here we save the results is files for better processing of files 
resultSummary = open("Experiment Result Summary.csv","w")
"""hidden_layer_nodes, seed, learn_rate, error, confusion_matrix_correctness"""
 
learn_rate = [0.05,0.1,0.3,0.5] # learning rates as asked in the question
hidden_layer_nodes = [1,2,5,10,20,30] # different hidden layer nodes
seed = [10,27,37,47] # seed counts.
# all different experiments are performed here and results are written into CVS file! 
for i in hidden_layer_nodes: 
    for s in seed: 
         for l in learn_rate:
            test = 1
            print "-----Seed="+str(s)+"-Learn-rate="+str(l)+"-Hidden-Layer="+str(i)+"-Test="+str(test)+"-----"
            net = mlp(train_in,train_tgt,i,s,outtype='softmax')
            error= net.earlystopping(train_in,train_tgt,valid_in,valid_tgt,l)
            confusion_matrix_correctness= net.confmat(test_in,test_tgt,s,i,l,test)
            L = '{0}, {1}, {2}, {3}, {4}, {5}\n'.format(i,s, l, test, error, confusion_matrix_correctness)
            resultSummary.write(L)
            
            if l == learn_rate[1] and s==seed[0]:
                test = 2
                print "-----Seed="+str(s)+"-Learn-rate="+str(l)+"-Hidden-Layer"+str(i)+"Test"+str(test)+"-----"
                net = mlp(train_in,train_tgt,i,s,outtype='softmax')
                error= net.earlystopping(train_in,train_tgt,valid_in,valid_tgt,l)
                confusion_matrix_correctness= net.confmat(test_in,test_tgt,s,i,l,test)
                L = '{0}, {1}, {2}, {3}, {4}, {5}\n'.format(i,s, l, test, error, confusion_matrix_correctness)
                resultSummary.write(L)

resultSummary.close()


# In[4]:


resultSummaryFile = np.genfromtxt("Experiment Result Summary.csv", delimiter=',')


# ## <font color=sienna>Observation</font>
# ### <font color=sienna>Best Cases:</font>
# 
# Here we observe that, 
# The Maximum Correctness Percentage is for the below mentioned cases:
# 
# When Seed = 10 and learning rate = 0.05 and hidden layer nodes = 20 : 82.6
# 
# When Seed = 27 and learning rate = 0.05 and hidden layer nodes = 20 : 81.7
# 
# When Seed = 37 and learning rate = 0.3 and hidden layer nodes = 20 : 82.8
# 
# When Seed = 47 and learning rate = 0.3 and hidden layer nodes = 20 : 81.1
# 
# The overall best case is when Seed = 37, learning rate = 0.3 and hidden layer nodes = 20 := 82.8
# 
# The Minimum Error is for the below mentioned cases:
# 
# When Seed = 10 and learning rate = 0.05 and hidden layer nodes = 10 : 133.1142047
# 
# When Seed = 27 and learning rate = 0.05 and hidden layer nodes = 20 : 139.3922055
# 
# When Seed = 37 and learning rate = 0.5 and hidden layer nodes = 20 : 129.8553536
# 
# When Seed = 47 and learning rate = 0.1 and hidden layer nodes = 20 : 128.830136
# 
# The overall least error is when Seed = 47, learning rate = 0.1 and hidden layer nodes = 20 := 128.83013604

# ## <font color=sienna>Result Discussion</font>
# 
# The training and testing of the data is now done. Now, we move on to plotting the values in the graph plot and experimenting if the results can be reproduced or not.

# #### <font color=sienna>Answer to Question 1.A:</font>
# Here two experinment are performed for each number of hidden layer nodes to check if the results are reproducable. As an answer we see that even if we have different seed numbers of same count of hidden layer neuron, we have the similar results.

# In[6]:


indices1 = np.where((resultSummaryFile[:,1]==seed[0]) & (resultSummaryFile[:,2]==learn_rate[1]) 
                   & (resultSummaryFile[:,3]==1))

indices2 = np.where((resultSummaryFile[:,1]==seed[0]) & (resultSummaryFile[:,2]==learn_rate[1]) 
                   & (resultSummaryFile[:,3]==2))
figureName = "Reproducibility Check for Error"

fig = pl.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.scatter(x=resultSummaryFile[indices1,0],y=resultSummaryFile[indices1,3],c="blue",marker='^',label="Test 1")
ax.scatter(x=resultSummaryFile[indices2,0],y=resultSummaryFile[indices2,3],c="red",marker='x',label="Test 2")
ax.set_title(figureName)
ax.set_xlabel('Number of Hidden Nodes')
ax.set_ylabel('Test File')
ax.legend(loc='best')
pl.show()


# In[7]:


figureName = "Reproducibility Check for Confusion Matrix Correctness"

fig = pl.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.scatter(x=resultSummaryFile[indices1,0],y=resultSummaryFile[indices1,5],c="blue",marker='^',label="Test 1")
ax.scatter(x=resultSummaryFile[indices1,0],y=resultSummaryFile[indices1,5],c="red",marker='x',label="Test 2")
ax.set_title(figureName)
ax.set_xlabel('Number of Hidden Nodes')
ax.set_ylabel('Confusion Matrix Correctness')
ax.legend(loc='best')
pl.show()


# In[8]:


print "Confusion matrix for hidden layer nodes of 30 as an example:"
print "\n"
print "Test 1:"
data = np.load('Experiment NHNode-30 LRate-0.1 Seed-10 Test-1.npy')
print(data)
print "\n"
print "Test 2:"
data = np.load('Experiment NHNode-30 LRate-0.1 Seed-10 Test-2.npy')
print(data)


# It is clear that after setting seed for the weights, both test produces same confusion matrix. It is true for all different number of hidden layer nodes. Also both test prodce same error and same correctness of confusion matrix. So the test results are reproducable.

# #### <font color=sienna>1.B. Experiment With Different Seed:</font>
# 

# In[9]:


"""hidden_layer_nodes, seed, learn_rate, error, confusion_matrix_correctness"""
print "\n Hidden Layer = "+ str(hidden_layer_nodes[0])+" Learning Rate = "+str(learn_rate[1])
indices1a = np.where((resultSummaryFile[:,0]==hidden_layer_nodes[0]) & (resultSummaryFile[:,2]==learn_rate[1]) 
                   & (resultSummaryFile[:,3]==1))

figureName = "Dependency of Confusion Matrix Correctness on Seed"

fig = pl.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.scatter(x=resultSummaryFile[indices1a,1],y=resultSummaryFile[indices1a,5],c="blue",marker='o')
ax.set_title(figureName)
ax.set_xlabel('Seed')
ax.set_ylabel('Confusion Matrix Correctness')
pl.show()


# In[10]:


"""hidden_layer_nodes, seed, learn_rate, error, confusion_matrix_correctness"""
indices11a = np.where((resultSummaryFile[:,1]==seed[0]) & (resultSummaryFile[:,3]==1))
indices21a = np.where((resultSummaryFile[:,1]==seed[1]) & (resultSummaryFile[:,3]==1))
indices31a = np.where((resultSummaryFile[:,1]==seed[2]) & (resultSummaryFile[:,3]==1))
indices41a = np.where((resultSummaryFile[:,1]==seed[3]) & (resultSummaryFile[:,3]==1))
figureName = "Dependency of Confusion Matrix Correctness on Seed"

fig = pl.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.scatter(x=resultSummaryFile[indices11a,0],y=resultSummaryFile[indices11a,5],c="blue",marker='o',label="Seed = 10")
ax.scatter(x=resultSummaryFile[indices21a,0],y=resultSummaryFile[indices21a,5],c="red",marker='x',label="Seed = 27")
ax.scatter(x=resultSummaryFile[indices31a,0],y=resultSummaryFile[indices31a,5],c="black",marker='^',label="Seed = 37")
ax.scatter(x=resultSummaryFile[indices41a,0],y=resultSummaryFile[indices41a,5],c="green",marker='*',label="Seed = 47")
ax.set_title(figureName)
ax.set_xlabel('Number of Hidden Nodes')
ax.set_ylabel('Confusion Matrix Correctness')
ax.legend(loc='best')
pl.show()


# In[11]:


data = np.load('Experiment NHNode-20 LRate-0.3 Seed-37 Test-1.npy')
print(data)
indices = np.where((resultSummaryFile[:,0]==hidden_layer_nodes[4]) & (resultSummaryFile[:,1]==seed[2]) 
                   & (resultSummaryFile[:,2]==learn_rate[2]) & (resultSummaryFile[:,3]==1))

print "\nHidden Layer = "+ str(hidden_layer_nodes[4])+" Learning Rate = "+str(learn_rate[2])+" Seed = "+str(seed[2])
print resultSummaryFile[indices,5]


# With the increase in the seed correctess of the confusion matrix increases. We also noticed for seed 1 to 5 correctness increases significantly but slows down after 5. It is also clear  from above diagram that maximum correctness is 82.8 occurs at seed = 37 and hidden layer nodes 20 for learning rate = 0.3

# In[12]:


figureName = "Dependency of Error on Seed"

fig = pl.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.scatter(x=resultSummaryFile[indices1a,1],y=resultSummaryFile[indices1a,4],c="blue",marker='o')
ax.set_title(figureName)
ax.set_xlabel('Seed')
ax.set_ylabel('Error')
pl.show()


# In[13]:


figureName = "Dependency of Error on Hidden Nodes"

fig = pl.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.scatter(x=resultSummaryFile[indices11a,0],y=resultSummaryFile[indices11a,4],c="blue",marker='o',label="Seed = 10")
ax.scatter(x=resultSummaryFile[indices21a,0],y=resultSummaryFile[indices21a,4],c="red",marker='x',label="Seed = 27")
ax.scatter(x=resultSummaryFile[indices31a,0],y=resultSummaryFile[indices31a,4],c="black",marker='^',label="Seed = 37")
ax.scatter(x=resultSummaryFile[indices41a,0],y=resultSummaryFile[indices41a,4],c="green",marker='*',label="Seed = 47")
ax.set_title(figureName)
ax.set_xlabel('Number of Hidden Nodes')
ax.set_ylabel('Error')
ax.legend(loc='best')
pl.show()


# In[14]:


print "\nHidden Layer = "+ str(hidden_layer_nodes[4])+" Learning Rate = "+str(learn_rate[1])+" Seed = "+str(seed[3])
indices = np.where((resultSummaryFile[:,0]==hidden_layer_nodes[4]) & (resultSummaryFile[:,1]==seed[3]) 
                   & (resultSummaryFile[:,2]==learn_rate[1]) & (resultSummaryFile[:,3]==1))
print resultSummaryFile[indices,4]


# With the increase in the seed, error decreases. We also noticed for seed 1 to 5 error decreases significantly but slows down after 5.It is also clear that minimum error is 128.83013604 occurs at seed = 47 and hidden layer nodes 20 and learning rate 0.1.

# #### <font color=sienna>Answer to Question 2 : Experiment With Different Learning Rate:</font>

# In[15]:


"""hidden_layer_nodes, seed, learn_rate, error, confusion_matrix_correctness"""

indices1b = np.where((resultSummaryFile[:,1]==seed[1]) & (resultSummaryFile[:,2]==learn_rate[0]) 
                   & (resultSummaryFile[:,3]==1))
indices2b = np.where((resultSummaryFile[:,1]==seed[1]) & (resultSummaryFile[:,2]==learn_rate[1]) 
                   & (resultSummaryFile[:,3]==1))
indices3b = np.where((resultSummaryFile[:,1]==seed[1]) & (resultSummaryFile[:,2]==learn_rate[2]) 
                   & (resultSummaryFile[:,3]==1))
indices4b = np.where((resultSummaryFile[:,1]==seed[1]) & (resultSummaryFile[:,2]==learn_rate[3]) 
                   & (resultSummaryFile[:,3]==1))
figureName = "Dependency of Confusion Matrix Correctness on Learning Rate at Seed = 27"

fig = pl.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.scatter(x=resultSummaryFile[indices1b,0],y=resultSummaryFile[indices1b,5],c="blue",marker='o',label="LR=0.05")
ax.scatter(x=resultSummaryFile[indices2b,0],y=resultSummaryFile[indices2b,5],c="red",marker='x',label="LR=0.1")
ax.scatter(x=resultSummaryFile[indices3b,0],y=resultSummaryFile[indices3b,5],c="black",marker='^',label="LR=0.3")
ax.scatter(x=resultSummaryFile[indices4b,0],y=resultSummaryFile[indices4b,5],c="green",marker='.',label="LR=0.5")
ax.set_title(figureName)
ax.set_xlabel('Number of Hidden Nodes')
ax.set_ylabel('Confusion Matrix Correctness')
ax.legend(loc='best')
pl.show()


# In[16]:


"""hidden_layer_nodes, seed, learn_rate, error, confusion_matrix_correctness"""
indices11 = np.where((resultSummaryFile[:,1]==seed[0]) & (resultSummaryFile[:,2]==learn_rate[0]) 
                   & (resultSummaryFile[:,3]==1))
indices21 = np.where((resultSummaryFile[:,1]==seed[1]) & (resultSummaryFile[:,2]==learn_rate[0]) 
                   & (resultSummaryFile[:,3]==1))
indices31 = np.where((resultSummaryFile[:,1]==seed[2]) & (resultSummaryFile[:,2]==learn_rate[0]) 
                   & (resultSummaryFile[:,3]==1))
indices41 = np.where((resultSummaryFile[:,1]==seed[3]) & (resultSummaryFile[:,2]==learn_rate[0]) 
                   & (resultSummaryFile[:,3]==1))
"""Learning 1"""
indices12 = np.where((resultSummaryFile[:,1]==seed[0]) & (resultSummaryFile[:,2]==learn_rate[1]) 
                   & (resultSummaryFile[:,3]==1))
indices22 = np.where((resultSummaryFile[:,1]==seed[1]) & (resultSummaryFile[:,2]==learn_rate[1]) 
                   & (resultSummaryFile[:,3]==1))
indices32 = np.where((resultSummaryFile[:,1]==seed[2]) & (resultSummaryFile[:,2]==learn_rate[1]) 
                   & (resultSummaryFile[:,3]==1))
indices42 = np.where((resultSummaryFile[:,1]==seed[3]) & (resultSummaryFile[:,2]==learn_rate[1]) 
                   & (resultSummaryFile[:,3]==1))
"""Learning 2"""
indices13 = np.where((resultSummaryFile[:,1]==seed[0]) & (resultSummaryFile[:,2]==learn_rate[2]) 
                   & (resultSummaryFile[:,3]==1))
indices23 = np.where((resultSummaryFile[:,1]==seed[1]) & (resultSummaryFile[:,2]==learn_rate[2]) 
                   & (resultSummaryFile[:,3]==1))
indices33 = np.where((resultSummaryFile[:,1]==seed[2]) & (resultSummaryFile[:,2]==learn_rate[2]) 
                   & (resultSummaryFile[:,3]==1))
indices43 = np.where((resultSummaryFile[:,1]==seed[3]) & (resultSummaryFile[:,2]==learn_rate[2]) 
                   & (resultSummaryFile[:,3]==1))
"""Learning 3"""
indices14 = np.where((resultSummaryFile[:,1]==seed[0]) & (resultSummaryFile[:,2]==learn_rate[3]) 
                   & (resultSummaryFile[:,3]==1))
indices24 = np.where((resultSummaryFile[:,1]==seed[1]) & (resultSummaryFile[:,2]==learn_rate[3]) 
                   & (resultSummaryFile[:,3]==1))
indices34 = np.where((resultSummaryFile[:,1]==seed[2]) & (resultSummaryFile[:,2]==learn_rate[3]) 
                   & (resultSummaryFile[:,3]==1))
indices44 = np.where((resultSummaryFile[:,1]==seed[3]) & (resultSummaryFile[:,2]==learn_rate[3]) 
                   & (resultSummaryFile[:,3]==1))
figureName = "Dependency of Confusion Matrix Correctness on Seed and Learning Rate"

fig = pl.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.scatter(x=resultSummaryFile[indices11,0],y=resultSummaryFile[indices11,5],c="blue",marker='o',label="Seed=10,LR=0.05")
ax.scatter(x=resultSummaryFile[indices21,0],y=resultSummaryFile[indices21,5],c="blue",marker='x',label="Seed=27,LR=0.05")
ax.scatter(x=resultSummaryFile[indices31,0],y=resultSummaryFile[indices31,5],c="blue",marker='^',label="Seed=37,LR=0.05")
ax.scatter(x=resultSummaryFile[indices41,0],y=resultSummaryFile[indices41,5],c="blue",marker='*',label="Seed=47,LR=0.05")

ax.scatter(x=resultSummaryFile[indices12,0],y=resultSummaryFile[indices12,5],c="black",marker='o',label="Seed=10,LR=0.1")
ax.scatter(x=resultSummaryFile[indices22,0],y=resultSummaryFile[indices22,5],c="black",marker='x',label="Seed=27,LR=0.1")
ax.scatter(x=resultSummaryFile[indices32,0],y=resultSummaryFile[indices32,5],c="black",marker='^',label="Seed=37,LR=0.1")
ax.scatter(x=resultSummaryFile[indices42,0],y=resultSummaryFile[indices42,5],c="black",marker='*',label="Seed=47,LR=0.1")

ax.scatter(x=resultSummaryFile[indices13,0],y=resultSummaryFile[indices13,5],c="red",marker='o',label="Seed=10,LR=0.3")
ax.scatter(x=resultSummaryFile[indices23,0],y=resultSummaryFile[indices23,5],c="red",marker='x',label="Seed=27,LR=0.3")
ax.scatter(x=resultSummaryFile[indices33,0],y=resultSummaryFile[indices33,5],c="red",marker='^',label="Seed=37,LR=0.3")
ax.scatter(x=resultSummaryFile[indices43,0],y=resultSummaryFile[indices43,5],c="red",marker='*',label="Seed=47,LR=0.3")

ax.scatter(x=resultSummaryFile[indices14,0],y=resultSummaryFile[indices14,5],c="green",marker='o',label="Seed=10,LR=0.5")
ax.scatter(x=resultSummaryFile[indices24,0],y=resultSummaryFile[indices24,5],c="green",marker='x',label="Seed=27,LR=0.5")
ax.scatter(x=resultSummaryFile[indices34,0],y=resultSummaryFile[indices34,5],c="green",marker='^',label="Seed=37,LR=0.5")
ax.scatter(x=resultSummaryFile[indices44,0],y=resultSummaryFile[indices44,5],c="green",marker='*',label="Seed=47,LR=0.5")
ax.set_title(figureName)
ax.set_xlabel('Number of Hidden Nodes')
ax.set_ylabel('Confusion Matrix Correctness')
ax.legend(loc='best')
pl.show()


# In[17]:


print "\n Hidden Layer = "+ str(hidden_layer_nodes[4])+" Learning Rate = "+str(learn_rate[2])+" Seed = "+str(seed[2])
data = np.load('Experiment NHNode-20 LRate-0.3 Seed-37 Test-1.npy')
print(data)
indices = np.where((resultSummaryFile[:,0]==hidden_layer_nodes[4]) & (resultSummaryFile[:,1]==seed[2]) 
                   & (resultSummaryFile[:,2]==learn_rate[2]) & (resultSummaryFile[:,3]==1))
print resultSummaryFile[indices,5]


# It is difficult to determine how confusion matrix correctness depend on learning rate. However, maximum correctness is 82.8 occurs at seed = 37, learning rate = 0.3 and hidden layer node 20.

# In[18]:


figureName = "Dependency of Error on Learning Rate at Seed = 27"

fig = pl.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.scatter(x=resultSummaryFile[indices1b,0],y=resultSummaryFile[indices1b,4],c="blue",marker='o',label="LR=0.05")
ax.scatter(x=resultSummaryFile[indices2b,0],y=resultSummaryFile[indices2b,4],c="red",marker='x',label="LR=0.1")
ax.scatter(x=resultSummaryFile[indices3b,0],y=resultSummaryFile[indices3b,4],c="black",marker='^',label="LR=0.3")
ax.scatter(x=resultSummaryFile[indices4b,0],y=resultSummaryFile[indices4b,4],c="green",marker='.',label="LR=0.5")
ax.set_title(figureName)
ax.set_xlabel('Number of Hidden Nodes')
ax.set_ylabel('Error')
ax.legend(loc='best')
pl.show()


# In[19]:


figureName = "Dependency of Error on Seed and Learning Rate"

fig = pl.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.scatter(x=resultSummaryFile[indices11,0],y=resultSummaryFile[indices11,4],c="blue",marker='o',label="Seed=10,LR=0.05")
ax.scatter(x=resultSummaryFile[indices21,0],y=resultSummaryFile[indices21,4],c="blue",marker='x',label="Seed=27,LR=0.05")
ax.scatter(x=resultSummaryFile[indices31,0],y=resultSummaryFile[indices31,4],c="blue",marker='^',label="Seed=37,LR=0.05")
ax.scatter(x=resultSummaryFile[indices41,0],y=resultSummaryFile[indices41,4],c="blue",marker='*',label="Seed=47,LR=0.05")

ax.scatter(x=resultSummaryFile[indices12,0],y=resultSummaryFile[indices12,4],c="black",marker='o',label="Seed=10,LR=0.1")
ax.scatter(x=resultSummaryFile[indices22,0],y=resultSummaryFile[indices22,4],c="black",marker='x',label="Seed=27,LR=0.1")
ax.scatter(x=resultSummaryFile[indices32,0],y=resultSummaryFile[indices32,4],c="black",marker='^',label="Seed=37,LR=0.1")
ax.scatter(x=resultSummaryFile[indices42,0],y=resultSummaryFile[indices42,4],c="black",marker='*',label="Seed=47,LR=0.1")

ax.scatter(x=resultSummaryFile[indices13,0],y=resultSummaryFile[indices13,4],c="red",marker='o',label="Seed=10,LR=0.3")
ax.scatter(x=resultSummaryFile[indices23,0],y=resultSummaryFile[indices23,4],c="red",marker='x',label="Seed=27,LR=0.3")
ax.scatter(x=resultSummaryFile[indices33,0],y=resultSummaryFile[indices33,4],c="red",marker='^',label="Seed=37,LR=0.3")
ax.scatter(x=resultSummaryFile[indices43,0],y=resultSummaryFile[indices43,4],c="red",marker='*',label="Seed=47,LR=0.3")

ax.scatter(x=resultSummaryFile[indices14,0],y=resultSummaryFile[indices14,4],c="green",marker='o',label="Seed=10,LR=0.5")
ax.scatter(x=resultSummaryFile[indices24,0],y=resultSummaryFile[indices24,4],c="green",marker='x',label="Seed=27,LR=0.5")
ax.scatter(x=resultSummaryFile[indices34,0],y=resultSummaryFile[indices34,4],c="green",marker='^',label="Seed=37,LR=0.5")
ax.scatter(x=resultSummaryFile[indices44,0],y=resultSummaryFile[indices44,4],c="green",marker='*',label="Seed=47,LR=0.5")
ax.set_title(figureName)
ax.set_xlabel('Number of Hidden Nodes')
ax.set_ylabel('Error')
ax.legend(loc='best')
pl.show()


# In[20]:


print "\nHidden Layer = "+ str(hidden_layer_nodes[4])+" Learning Rate = "+str(learn_rate[1])+" Seed = "+str(seed[3])
indices = np.where((resultSummaryFile[:,0]==hidden_layer_nodes[4]) & (resultSummaryFile[:,1]==seed[3]) 
                   & (resultSummaryFile[:,2]==learn_rate[1]) & (resultSummaryFile[:,3]==1))
print resultSummaryFile[indices,4]


# It is difficult to determine how error depend on learning rate. However, minimum error is 128.83013604 occurs at seed = 47, learning rate = 0.1 and hidden layer node 20.

# ## <font color=sienna>Conclusion</font>
# 
# Based on the data tested using hidden layer nodes of 1,2,5,10,20, and 30 training over the first 1000 examples using a variety of learning rates the best results occured on seed = 37, learning rate of 0.3, and hidden layer node 20. The minimum error was seen though in seed =47, learning rate of 0.1 and hidden layer node 20. The worst results occured on seed = 27, learning rate of 0.3 and hidden layer node 1. It is difficult to track any real correlation between the learning rates and the error of the training, however it is observed that too few hidden layer nodes leads to a high error more often than too many.

# These are the files created while running the experiment based on all the values we used.
# ![snip.PNG](attachment:snip.PNG)

# In[ ]:




