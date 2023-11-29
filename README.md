# Extracellular-Action-Potential-Classification
Machine learning code implemented for classification of action potentials in electrode readings from cortical neurons. 

# Overview

This project focuses on the classification of extracellular action potentials (spikes) 
found in electrode readings from the cortical region of the brain. Two sets of readings 
were supplied: in the first set, the reference set, the location and class of the spikes 
were provided; however, in the second set of readings, the submission set, only the raw 
data was given. The first set was used to train the classification system and the second
set was used by the lecturer to evaluate the classification system against their labelled 
version of the submission set which only they possessed. The reference data was separated 
into training, validation and testing sets. Within the two given datasets, there were 
five different types of spikes that were produced by five different types of neuron. 
Each type of spike had a different morphology which allowed it to be distinguished from 
the others but in some cases the spikes overlapped which complicated the classification.
Furthermore, both data sets provided had significant levels of noise which also increased 
the complexity of the classification. 

The classification process consisted of three steps:

1. data pre-processing - raw data was filtered to remove noise and make the spike peaks easier to detect

2. peak detection - algorithm implemented to search data for peaks and note the positions at which they occurred

3. classification - detected spikes analysed categorised into classes 

# Project Structure

1. Functions.py - contains all the required functions to run the classifier models and genetic algorithm including filtering and peak detection code

2. MLP Method.py - implements, trains and tests multilayer perceptron classifier using reference data set 

3. KNN Method.py - performs feature extraction on reference data and implements, trains and tests K-Nearest Neighbour classifier 

4. GeneticAlgorithm.py - implements a genetic algorithm to optimise MLP classifier

# Author 

Louis Chapo-Saunders
