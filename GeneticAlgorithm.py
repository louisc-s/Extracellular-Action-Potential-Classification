from random import randint , random, uniform
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import copy
import scipy.io as spio
from keras import Sequential, optimizers
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from Functions import CreateDataSets, noise_filter, makeCSV ,get_inputs_targets, MatchClassIndex, findpeaks


#create individuals from randomly generated 
# MLP paramters 
def createIndividual():
    epoch = randint(20,300) 
    neuronnumber = randint(10,300)
    learningrate = uniform(0.001,0.1)
    batchsize = randint(10,300)
    individual = [epoch,neuronnumber, batchsize, learningrate]
    return individual

#create population from multiple individuals
def createPopulation(populationsize):
    population = []
    for i in range(0,populationsize):
        ind = createIndividual()
        population.append(ind)
    return population

#evaluate fitness of individuals by running MLP and evaluating accuracies 
def fitness(ind):

        #load file and extract training data 
        mat = spio.loadmat('training.mat', squeeze_me=True)
        data = mat['d']
        Index = mat['Index']
        Class = mat['Class']

        filt_data = noise_filter(data) #filter noise from data with band-pass filter 

        PredIndex, _ = findpeaks(filt_data) #detect spike peaks within filtered data

        #assign class to detected spikes
        DetIndex , DetClass = MatchClassIndex(PredIndex,Index,Class) 

        #create training, validation and test data sets from class and index data
        train_data,vali_data, test_data = CreateDataSets(DetClass,DetIndex)

        #generate CSV files containing peak data points and class category
        #for each type of dataset
        makeCSV(train_data,filt_data,1)
        makeCSV(vali_data,filt_data,2)
        makeCSV(test_data,filt_data,3)

        #read CSV files, extract data held in them 
        # and process the data so that it in a suitable form to be input
        # into the MLP
        train_inputs, train_targets = get_inputs_targets(1)
        val_inputs, val_targets = get_inputs_targets(2)
        test_inputs, test_targets = get_inputs_targets(3)

        #define number of input and output nodes for MLP
        input_nodes = len(train_inputs[0])
        output_nodes = 5

        # define model
        model = Sequential()
        # create input layer and first hidden layer
        model.add(Dense(ind[1], input_dim=input_nodes))
        # create output layer
        model.add(Dense(output_nodes, activation='softmax'))
        # define loss function and optimisation algorithm
        model.compile(optimizer= optimizers.Adam(learning_rate=ind[3]), loss='categorical_crossentropy', metrics=['accuracy'])
        # train the MLP with the training data and validate it with the validation data
        history = model.fit(train_inputs, train_targets, validation_data=(val_inputs,val_targets), epochs= ind[0], batch_size= ind[2], verbose=0)

        #test the MLP using test data
        loss, acc = model.evaluate(test_inputs, test_targets, verbose=0)

        #return the fitness (accuracy) of the individual
        return acc

#select fit individuals for parent population
def select(population,target):

    #define selection parameters
    retain = 0.2
    random_select = 0.05

    #creates list with fitness of individual for all individuals in population
    graded = [ (fitness(x), x) for x in population]
    a = graded
    a.sort(reverse = True)

    #creates sorted list of individuals from the ones that produce the most
    #accurate classification to those which produce the least accurate
    graded = [ x[1] for x in a]

    #creates sorted list of fitness values (accuracies)
    accuracy = [x[0] for x in a]
    
    #calculate summed fitness for population
    sum = 0
    for datum in accuracy:
        sum = sum + datum

    score = sum/len(population) #calculate average population fitness
    fit_val = a[0][0] #store fitness value of fittest indvidual
    fit_param = a[0][1] #store parameters of fittest individual

    #check if solution that matches target accuracy has been found 
    if a[0][0] >= target:
        print("solution found", a[0][1])
    
    #changes length of graded list
    retain_length = int(len(graded)*retain)
    #keeps highest perfroming indviduals within a certain proprotion of list to use as parents 
    parents = graded[:retain_length]

    #randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    return parents, fit_val, score, fit_param

#create children from parent individuals 
def crossover(parents,population):
    #define crossover parameter
    crossover =0.9
    #crossover parents to create children
    parents_length = len(parents)
    desired_length = len(population) - parents_length #calculate required number of children
    children = []

    while  desired_length > len(children):
        if crossover > random():
            #select two indivduals within parents list and assign male or female 
            male = randint(0, parents_length-1) 
            female = randint(0, parents_length-1)
             #create child using binary uniform crossover of male and female individuals 
            if male != female: #check male and female are not the same individual
                #create child variables 
                child1 = []
                child2 = []
                CHILD1 = []
                CHILD2 = []
                #convert first three male elements into binary
                male = (parents[male])
                mbin1 = bin(male[0])
                mbin2 = bin(male[1])
                mbin3 = bin(male[2])
                 #convert first three female elements into binary
                female = (parents[female])
                fbin1 = bin(female[0])
                fbin2 = bin(female[1])
                fbin3 = bin(female[2])
                #discard initial "ob" string section produced during conversion
                bmale1 = mbin1[2:]
                bmale2 = mbin2[2:]
                bmale3 = mbin3[2:]
                bfemale1 = fbin1[2:]
                bfemale2 = fbin2[2:]
                bfemale3 = fbin3[2:]
                #group elemnts into a binary individuals
                BMALE = [bmale1,bmale2,bmale3]
                BFEMALE = [bfemale1,bfemale2,bfemale3]
                #iterate through elements in binary indviduals 
                for m,f in zip(BMALE,BFEMALE):
                    #check if male and female elements are the same length
                    #and adjust them to be if they are not
                    if len(m) > len(f):
                        f = f.zfill(len(m))
                    if len(f) > len(m):
                        m = m.zfill(len(f))
                    child1 = []
                    child2 = []
                    #randomly crossover the binary sequencues of
                    # the male and female elements to produce two 
                    #children
                    for i in range(len(m)):
                                    
                                    if randint(0, 1):
                                        child1.append(m[i])
                                        child2.append(f[i])
                                    else:
                                        child1.append(f[i])
                                        child2.append(m[i])
                    #convert binary child element into decimal value                
                    c1 = ''.join(child1) 
                    c2 = ''.join(child2)
                    CHILD1.append(int((c1),2))
                    CHILD2.append(int((c2),2))
                #ensure no element of the child = 0
                #to prevent errors in the MLP
                for i in range(0,len(CHILD1)):
                    if CHILD1[i] ==0:
                        CHILD1[i] =1
                    if CHILD2[i] ==0:
                        CHILD2[i] = 1    

                #combine male and female learning rates 
                #to produce last element of each child
                newlr1 = (male[3]*0.7+ female[3]*0.3)
                newlr2 = (male[3]*0.3+ female[3]*0.7)
                #append combined learning rates onto
                # the child individuals
                if randint(0, 1):
                    CHILD1.append(newlr1)
                    CHILD2.append(newlr2)
                else: 
                    CHILD1.append(newlr2)
                    CHILD2.append(newlr1)
                #add first child to children list
                children.append(CHILD1)
                #check whether desired number of children has 
                #been reach before adding second child
                if desired_length > len(children):
                    children.append(CHILD2) 

        else: 
            #select certain indvidual within parents list to be cloned 
            asexual = randint(0, parents_length-1)
            clone = parents[asexual]   
            children.append(clone)

    parents.extend(children) #add children to parents 
    return parents


#mutate some individualsfor each individual if the 
# random number generator gives number less than mutate value 
#the a random number is added to a gene within the individual
def mutation(parents):
    mutate = 0.12 #define mutation parameter 

    for individual in parents:
        if mutate > random():
            #determine which element of the individual is
            #to be mutated 
            pos_to_mutate = randint(0, len(individual)-1) 
            #mutate each element differently depending on the 
            # which MLP parameter it is
            if pos_to_mutate == 0:
                imin = 1
                imax = 300
                individual[pos_to_mutate] = randint(imin, imax)
            elif pos_to_mutate == 1:
                imin = 5
                imax = 90
                individual[pos_to_mutate] =  randint(imin, imax)
            elif pos_to_mutate == 2:
                imin = 10
                imax = 300
                individual[pos_to_mutate] =  randint(imin, imax)
            elif pos_to_mutate == 3:
                imin = 0.001
                imax = 0.01
                individual[pos_to_mutate] =  uniform(imin, imax)
   
    return parents

generations = 50 #define number of generations
populationsize = 60 #define population size
target = 0.99 #define target accuracy to optimse to

population = createPopulation(populationsize) #create population of parameter individuals
top_results = [] #list to store top indviduals 
fitness_history = [] #list to store the avergae fitness of the population for each generation


#code to run genetic algorithm for set number of generations 
for i in range(generations):
    gen_count = i
    print("generations:", gen_count)
    parents, Accuracy,score, param = select(population, target) #rate individuals and select parents 
    top_results.append(Accuracy) #record highest fitness value of the generation
    fitness_history.append(score) #record average fitness of population for this generation

    print("top:",Accuracy,"individual:", param, "average:",score)
    #stop the program if target accuracy has been achieved 
    if Accuracy > target:
        exit()
    #stop the program if evolution has stagnated 
    if gen_count > 5:
        if fitness_history[i] == fitness_history[i-1]:
            if fitness_history[i-1] == fitness_history[i-2]:
                if fitness_history[i -2] == fitness_history[i-3]:
                    if fitness_history[i -3] == fitness_history[i-4]:
                        if fitness_history[i -4] == fitness_history[i-5]:
                            exit()
    parents = crossover(parents,population) #create children from parents
    population = mutation(parents) #mutate parents and children and update population
    
print(top_results,"\n")
print(fitness_history)










