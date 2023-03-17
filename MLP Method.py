

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy.lib.function_base import copy
import scipy.io as spio
from keras import Sequential, optimizers
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from Functions import CreateDataSets, findpeaks, noise_filter, makeCSV ,get_inputs_targets, MatchClassIndex, getinputs, makeSubCSV, subfindpeaks


#############TRAINING#################

#load file and extract training data 
mat = spio.loadmat('training.mat', squeeze_me=True)
data = mat['d']
Index = mat['Index']
Class = mat['Class']

filt_data = noise_filter(data)  #filter noise from data with band-pass filter 
PredIndex, _ = findpeaks(filt_data) #detect spike peaks within filtered data
#assign class to detected spikes
DetIndex , DetClass = MatchClassIndex(PredIndex,Index,Class)
#create training, validation and test data sets from class and index data
train_data,vali_data, test_data = CreateDataSets(DetClass,DetIndex)

#Generate CSV file containing peak data points and class category
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
# create Input layer and first hidden layer
model.add(Dense(3, input_dim=input_nodes))
# create Output layer
model.add(Dense(output_nodes, activation='softmax'))
# define loss function and optimisation algorithm
model.compile(optimizer= optimizers.Adam(learning_rate= 0.0056), loss='categorical_crossentropy', metrics=['accuracy'])
# train the MLP with the training data and validate it with the validation data
history = model.fit(train_inputs, train_targets, validation_data=(val_inputs,val_targets), epochs=181, batch_size=363, verbose=0)#batch size was 128

######## TESTING##########

# test the MLP using test data
loss, acc = model.evaluate(test_inputs, test_targets, verbose=1)
# print accuracy of MLP 
print(acc)

#test MLP again using model.predict function so that performance metrics can be generated 

#create lists to store actual and predicted classes
# to allow performance metrics to be generated 
Actual_Class = []
Predicted_Class = []

for input, target in zip(test_inputs, test_targets):
    # predict classes of input spikes and append to list
    prediction = model.predict(np.asarray([input])) 
    Predicted_Class.append(np.argmax(prediction)+1) 
    # append actual class of input spike to list
    Actual_Class.append(np.argmax(target)+1)
    
#create and display confusion matrix
cm = confusion_matrix(Actual_Class, Predicted_Class)
display = "Class 1","Class 2","Class 3","class 4","Class 5"
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display)
disp.plot(cmap="Blues")

#create classification report 
cr = classification_report(Actual_Class,Predicted_Class)
print(cr)
plt.show()


########### SUBMISSION DATA TESTING ################

#load file and extract submission data 
mat = spio.loadmat('submission.mat', squeeze_me=True)
subdata = mat['d']

filt_sub = noise_filter(subdata) #filter submission data with band-pass filter to remove noise

SubPredIndex, _ = subfindpeaks(filt_sub) #detect spike peaks within the submission data 

#Generate CSV file containing peak data points 
#from the submission data set
makeSubCSV(SubPredIndex,filt_sub)

#read submission CSV file, extract data held in it 
# and process the data so that it in a suitable form to be input
# into the MLP
Subinputs = getinputs()

#convert input data into a set of arrays
Subinputs = np.array(Subinputs)
#use the MLP to assign a class to each detected spike from the data - 
# predictions are returned  in the form of an array of probabilties 
# that determine the chance that a paritcular spike is of a particular class 
SubPredClass = model.predict(Subinputs)
#pick the highest probability from the prediction array and assign
# the class asscociated with this probability to detected spike. Do 
# this for every dectected spike
SubPredClass = [np.argmax(targets) +1 for targets in SubPredClass]

# create arrays for the detected spike locations and asscoiated predicted classes
Index = np.array(SubPredIndex[1:]) #discard index for initial noise peak
Class = np.array(SubPredClass)
#spio.savemat("13783.mat", {"Index":Index, "Class":Class}) # save these arrays in a matlab file (uncomment to save new array)










