#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Introduction:   
    
    This code is cut-out from our workflow to show the neural networks generation and the usage of the relevance algorithm.
    It is (sadly) not intended to work out of the box, since we can't provide the dataset.
    You will have to bring your own data and then fit the network to your dataset. 
    You also need the toolbox 'keras' and 'tensorflow'. A GPU is not neccessary, the calculations run there only 20% faster
    
    If you have any questions about this code, I am willing to help. Please email me at
    tobias.de.taillez@uni-oldenburg.de
    or
    irazari@zae-ne.de
    
    Tobias de Taillez, Sept.2017
    Universität Oldenburg, Germany
'''

import gc
#import keras
from keras.models import Sequential, Model
#from keras.optimizers import SGD
from keras.regularizers import l2,l1 #, activity_l2,activity_l1
from keras.layers import Dense, Input, Dropout
from keras.engine.topology import Merge
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.callbacks import ModelCheckpoint


import scipy.io as io
import numpy as np
import sys
import time





numHiddenLayer=1
n_hidden=2 #Num of hidden neurons
numSampleContext=16 #training prediction window
loss=2 # correlation based loss function

numChan=len(chans)
netSize=28 #420 ms
dataLength=int(dataFs*50*60*dataAmount) #hz * 60 seconds *50 minutes of Data
    
###Functions 
   
def corr_loss(act,pred):          #Custom tailored loss function that values a high correlation. See our Paper for details 
    cov=(K.mean((act-K.mean(act))*(pred-K.mean(pred))))
    return 1-(cov/(K.std(act)*K.std(pred)+K.epsilon()))

    
def create_base_network(net_input_dim,dropout,bias,initStyle,l1Reg,l2Reg):
    '''Base network to be shared (eq. to feature extraction).
    net_input_dim: size of subnet in TimeSamples
    '''
    seq = Sequential() #Feed Forward Model
    if numHiddenLayer==0:    
        seq.add(Dense(1,activation='linear',input_dim=net_input_dim,b_regularizer=l1(l1Reg),W_regularizer=l1(l1Reg),init=initStyle,bias=False))    
    else:
        seq.add(Dense(n_hidden,activation='tanh',input_dim=net_input_dim,b_regularizer=l1(l1Reg),W_regularizer=l1(l1Reg),init=initStyle,bias=bias)) #Input Layer
        if dropout>0.0:            
            seq.add(Dropout(dropout))
        for layer in range(1,numHiddenLayer):
            seq.add(Dense(np.max([np.round(n_hidden/(layer+1)),1]),activation='tanh',b_regularizer=l2(l2Reg),W_regularizer=l2(l2Reg),init=initStyle,bias=bias))               
            if dropout>0.0:            
                seq.add(Dropout(dropout))
        seq.add(Dense(1,activation='linear',b_regularizer=l2(l2Reg),W_regularizer=l2(l2Reg),init=initStyle,bias=bias)) #Output Layer. 1 Neuron
    return seq


def get_activations1(model, layer, X_batch): #See get_interpret_new()
    if layer==-1: #InputLayer returns data itself
        return [X_batch]
    else: #Output of Layer X
        get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
        activations = get_activations([X_batch,0])
        return activations  
        

def get_interpret_new(model, X_batch,targetData,correctness, num_time_samples=0,bias=False,dropout=0.0):
    '''This is a self-tailored Version of Sturm et al 2016 Relevance algorithm
    Sadly there is NO easy-to-use Version for all possible network layouts. This is mainly due to the weights that play a role in 
    relevance calculation and the weights that DONT. These have to be deleted from the weight matrix. 
    Example:
        For a given Layer WITH Bias values for each neuron, these bias weight-layer have to be deleted since 
        in the mathematical equation only the activation of a neuron plays a role. We get this activation from get_activations1(model,layer,Data)
        and it already includes the effect of the bias layer.
        Sme for Dropout-layers. These appear in the activations-List but have to be deleted. In general you dont want any dropout occuring while testing/evaluation! 
        So generate your evaluation-network without dropout and then copy the weights from your trained network to this.
    '''
    weights=model.get_weights()
            
    # total number of layers (input + output + hidden layers)
    numLayers=len(weights)
    

    if dropout>0.0:
        activations=list()
        if bias:
            for layer in range(-1,len(weights)-numHiddenLayer+numHiddenLayer-1):
                activations.append(get_activations1(model,layer,X_batch)[0])
        else:                    
            for layer in range(-1,len(weights)+numHiddenLayer):
                activations.append(get_activations1(model,layer,X_batch)[0])
                
        activationLength=len(activations)                
        for layer in reversed(range(2,activationLength,2)):
            del activations[layer]

#        If Bias, delete Bias-Layers from Weights
    if bias==True:            
        for layer in reversed(range(1,numLayers,2)):
            del weights[layer]


    numLayers=len(weights)


    if (num_time_samples == 0)or(num_time_samples==-1):
        num_time_samples = activations[0].shape[0]
    # calculated relevance, list of arrays:
    # one 2d-array for each layer, indexed as (time-sample, node) (same as activations)
    relevance = list()
    
    # relevance of the output nodes
#        outputRelevance=np.ones((num_time_samples, num_output_nodes)) / num_output_nodes
    outputRelevance=correctness
#        outputRelevance=1/(1+np.abs(activations[-1]-targetData))
    relevance.append(outputRelevance)
    
    
    # determine relevance per layer
    for layer in reversed(range(numLayers)):
        num_lower_nodes = activations[layer].shape[1]
        num_upper_nodes = activations[layer + 1].shape[1]
        # initialize relevance of current layer to zeros
        layer_relevance = np.zeros((num_time_samples, num_lower_nodes))
        
        # determine relevance of nodes in lower layer
        for upper_node in range(num_upper_nodes):
            upper_activation = activations[layer + 1][:num_time_samples, upper_node]
            upper_relevance = relevance[-1][:, upper_node]
            # calculate contribution of the current upper node for all time samples
            upper_contribution = upper_relevance / upper_activation
            # sum up for all the nodes in the lower layer for all time samples:
            # contribution of the upper node * weights from the current upper node to all lower nodes
            # this uses matrix multiplication:
            # upper_contribution as a column-vector (:,1) * weights as a row-vector (1,:)
            # see https://de.wikipedia.org/wiki/Matrizenmultiplikation#Spaltenvektor_mal_Zeilenvektor
            upper_contribution = upper_contribution.reshape(-1, 1)
            weight = weights[layer][:, upper_node].reshape(1, -1)
            layer_relevance += np.matmul(upper_contribution, weight)
        # lower activation can be factored out of all the terms within the sum
        layer_relevance *= activations[layer][:num_time_samples]
        # layer relevance complete
        relevance.append(layer_relevance)
    # reverse relevance to match the layer-order of activations, i.e. 0 is the input layer
    relevance.reverse()
    return relevance

#### Main Code



##########Build Net########
baseNet=create_base_network(netSize*numChan,dropout,bias,initStyle,l1Reg,l2Reg) #Create the base network that is to be used multiple times (numSampleContext denotes the number of usages)

if numSampleContext==1:
    inputTensor=Input(shape=(netSize*numChan,))
    processed=baseNet(inputTensor)
    out=processed
    modelA = Model(input=inputTensor, output=out)
else:
    inputTensor=[Input(shape=(netSize*numChan,)) for i in range(numSampleContext)] #Generate list of input tensors
    processedTensor=[baseNet(inputTensor[i]) for i in range(numSampleContext)]    #Generate list of baseNet applications to their respective input tensor
    lay=Merge(mode='concat')(processedTensor) #Merge multiple baseNet instances
    modelA = Model(input=inputTensor, output=lay) #create Model
if loss==1: # Compilation
    modelA.compile(optimizer='Nadam',loss='mse')  #mse Model
elif loss==2:
    modelA.compile(optimizer='Nadam',loss=corr_loss)  #CorrLoss Model

              
checkLow = ModelCheckpoint(filepath=workingDir+'weights_lowAcc'+str(GPU)+'.hdf5', verbose=0, save_best_only=True,mode='min',monitor='val_loss') #Checkpoints for the lowest achieved evaluation loss       
early = EarlyStopping(monitor='val_loss',patience=earlyPatience, mode='min') #Daemon to stop before the maximum number of epochs is reached. It checks if the validation loss did not decrese for the last 'earlyPatience' trials

modelA.fit_generator((dataGenerator),nb_epoch=500,verbose=1,max_q_size=1,nb_worker=3,pickle_safe=True,validation_data=validationGenerator,callbacks=[checkLow,early])
'''
Fitting process
You have to write a data generator function (see keras manual) that returns a tupel of (EEGData, Envelope)
see modelA.input_shape and modelA.output_shape for details

'''

modelA.load_weights(filepath=workingDir+'/weights_lowAcc'+str(GPU)+'.hdf5') #Load the weights that worked best on the evaluation set


gc.collect() #collect garbage
    



baseNetA=create_base_network(netSize*numChan,0,bias,initStyle,0,0) #Create BaseNet and load the weights from Training
baseNetA.set_weights(modelA.get_weights())




#calculate input neuron relevance for each of the input neurons
data,envA,envU=''' insert Test Data Set here'''
envA+=np.random.randn(envA.shape[0],envA.shape[1])*0.000001

predA=baseNetA.predict_on_batch(data) #Use the net to predict the test set
numSampleContext=16
if neuronWeightingFlag: #calculate the input neuron relevance per sample for the test net
    corrMatrixCorrect=np.full((predA.shape[0],numSampleContext),np.nan)
    for samp in range(predA.shape[0]-numSampleContext):
        corrMatrixCorrect[samp:samp+numSampleContext,np.mod(samp,numSampleContext)]=np.corrcoef(predA[samp:samp+numSampleContext].T,envA[samp:samp+numSampleContext,0,None].T)[0][1]
    correctness=np.nanmean(corrMatrixCorrect[:predA.shape[0]-numSampleContext],axis=1).reshape((predA.shape[0]-numSampleContext,1))
    relevanceNeurons=get_interpret_new(baseNet,data[:-numSampleContext,:],envA[:-numSampleContext,0,None],correctness,-1,bias,dropout)
    neuronWeighting=np.median(np.squeeze(relevanceNeurons[0][np.where(correctness[:,0]>0),:]),axis=0)
    neuronWeightingStd=np.std(np.squeeze(relevanceNeurons[0][np.where(correctness[:,0]>0),:]),axis=0)
    


#Evaluate performance for different analysis window lengths
blockRange=np.asarray([dataFs*60,dataFs*10,dataFs*5,dataFs*3,dataFs*2.5,dataFs*2,dataFs*1,dataFs/2]) #analysis window lengths

corrResults=np.zeros((9,len(blockRange)))

for blockLengthIterator in range(len(blockRange)):
    t=time.time()
    blockLength=int(blockRange[blockLengthIterator])
    corrA=np.asarray(range(0,envA.shape[0]-blockLength))*np.nan
    corrU=np.asarray(range(0,envU.shape[0]-blockLength))*np.nan

    for block in range(corrA.shape[0]): #for a specific analysis window, run trough the test set prediction and correlate with attended and unattended envelope
        corrA[block]=np.corrcoef(envA[block:block+blockLength].T,predA[block:block+blockLength].T)[0][1]
        corrU[block]=np.corrcoef(envU[block:block+blockLength].T,predA[block:block+blockLength].T)[0][1]   

    corrResults[0,blockLengthIterator]=np.nanmean(corrA)
    corrResults[1,blockLengthIterator]=np.nanstd(corrA)
    corrResults[2,blockLengthIterator]=np.nanmean(corrU)        
    corrResults[3,blockLengthIterator]=np.nanstd(corrU)
    corrResults[4,blockLengthIterator]=np.nanmean(np.clip(corrA,-1,1)>np.clip(corrU,-1,1)) # Values the networks decision. 1 denotes "correct" zero "wrong". Averages also over the complete test set. This result the networks accuracy!
    accuracy=corrResults[4,blockLengthIterator]
    corrResults[5,blockLengthIterator]=(np.log2(2)+accuracy*np.log2(accuracy)+(1-accuracy)*np.log2((1-accuracy+0.00000001)/1))*dataFs/blockLength*60       
    corrResults[6,blockLengthIterator]=blockLength
    corrResults[7]=trainPat[0] #Which participant is evaluated
    corrResults[8]=startPointTest #At which time point did the evaluation/test set started

#Save Results
if neuronWeightingFlag:
    io.savemat('backTell'+str(GPU)+'.mat',{'corrResults':corrResults,'neuronWeighting':neuronWeighting,'neuronWeightingStd':neuronWeightingStd})
else:
    io.savemat('backTell'+str(GPU)+'.mat',{'corrResults':corrResults})

