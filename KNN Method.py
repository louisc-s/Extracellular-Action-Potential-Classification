import scipy.io as spio
import numpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from Functions import noise_filter, CreateDataSets, findpeaks, makeCSV, MatchClassIndex

#################TRAINING###################

#load training data set and create class and index variables
mat = spio.loadmat('training.mat', squeeze_me=True)
data = mat['d']
Index = mat['Index']
Class = mat['Class']

filt_data = noise_filter(data) # filter noisy data with band-pass filter 

PredIndex, _ = findpeaks(filt_data)# find spike peaks in the filtered data set

# match detcted spike peaks indexes with 
# true spike indexes and their associated classes
DetIndex , DetClass = MatchClassIndex(PredIndex,Index,Class)

# create training, validation and test data sets from class and index data
train_data,vali_data,test_data = CreateDataSets(DetClass,DetIndex)

#generate CSV files containing peak data points and 
# class category for each type of dataset
makeCSV(train_data,filt_data,1)
makeCSV(vali_data,filt_data,2)
makeCSV(test_data,filt_data,3)


# load the training, validation and test data sets
train = numpy.loadtxt('train_data.csv', delimiter=',')
test = numpy.loadtxt('test_data.csv', delimiter=',')
validate = numpy.loadtxt('validate_data.csv', delimiter=',')

# separate labels from all data sets
train_data = train[:, 1:]
train_labels = train[:, 0]
test_data = test[:, 1:]
test_labels = test[:, 0]
validate_data = validate[:, 1:]
validate_labels = validate[:, 0]

pca = PCA(n_components = 15)# select number of components to extract

pca.fit(train_data)# fit the model to the training data

# extract the principal components from the training data
train_ext = pca.fit_transform(train_data)

# reduce the dimensionality of the test and validation data 
# using the same components
test_ext = pca.transform(test_data)
validate_ext = pca.transform(validate_data)

# normalise the data sets
min_max_scaler = MinMaxScaler()
train_norm = min_max_scaler.fit_transform(train_ext)
test_norm = min_max_scaler.fit_transform(test_ext)
validate_norm = min_max_scaler.fit_transform(validate_ext)

# instaniate KNN classifier
knn = KNeighborsClassifier(n_neighbors= 35, p= 4)
knn.fit(train_norm, train_labels)

# feed validation data into the classifier to get the predictions
vpred = knn.predict(validate_norm)

# evaulate claissfier performance on test data set
validscore = []
for i, sample in enumerate(validate_data):
    # check if the KNN classification was correct
    if round(vpred[i]) == test_labels[i]:
        validscore.append(1)
    else:
        validscore.append(0)
pass

# calculate the overall accuracy of classifier on validation set
scorecard_array = numpy.asarray(validscore)
valid_acc = scorecard_array.sum() / scorecard_array.size * 100

############ TESTING #################

# feed test data into the classifier to get the predictions
tpred = knn.predict(test_norm)

# evaulate claissfier performance on test data set
testscore = []
for i, sample in enumerate(test_data):
    # check if the KNN classification was correct
    if round(tpred[i]) == test_labels[i]:
        testscore.append(1)
    else:
        testscore.append(0)
pass

# calculate the overall accuracy of classifier on test set
scorecard_array = numpy.asarray(testscore)
test_acc = scorecard_array.sum() / scorecard_array.size * 100
print(test_acc)

#create and display confusion matrix
cm = confusion_matrix(test_labels, tpred)
display = "Class 1","Class 2","Class 3","class 4","Class 5"
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display)
disp.plot(cmap="Blues")

#create classification report 
cr = classification_report(test_labels,tpred)
print(cr)
plt.show()

