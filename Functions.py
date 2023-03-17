import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from scipy.signal import butter, find_peaks, sosfilt
from sklearn.preprocessing import MinMaxScaler
from random import shuffle
import csv

# function to filter data using band-pass filter
def noise_filter(inputsignal):
    #define filter characteristics
    co = butter(3,[0.00156,0.35], btype = 'bandpass', output = 'sos')
    #implement filter
    filtered = sosfilt(co,inputsignal)
    return filtered

# function to detect spike peaks within the reference data
def findpeaks(data):
    peaks, _ = find_peaks(data, height = 1, width = 9, distance = 10, prominence = 1)
    return peaks, _

# function to detect spike peaks within the submission data
def subfindpeaks(data):
    peaks, _ = find_peaks(data, height = 1.4, width = 9, distance = 10, prominence = 1)
    return peaks, _

# function to normalise dataset with MinMaxScaler
#so that all values fall between 0 and 1
def normaliser(data):
    scaler = MinMaxScaler()
    normadata = np.array(scaler.fit_transform(data))
    return normadata

#function to separate data into training, validation 
#and test subsets in the ratio 0.7:0.15:0.15 
def split(Class, Index):

    #extrat 70% of classes and indexes for training set
    trainclass = Class[:int(len(Class)*0.7)]
    trainindex = Index[:int(len(Class)*0.7)]
    train = list(zip(trainclass,trainindex))

    #extract 15% of classes and indexes for validation data
    splitclist = Class[len(trainclass):]
    splitilist = Index[len(trainclass):]
    midpoint = len(splitclist) //2
    valclass = splitclist[:midpoint]
    valindex = splitilist[:midpoint]
    validate = list(zip(valclass,valindex))

    #extract remaining 15% of classes and indexes for test data
    testclass = splitclist[midpoint:]
    testindex = splitilist[midpoint:]
    test = list(zip(testclass,testindex))

    return train, validate, test

# function to merge data from different spike classes to
# ensure each type of data subset has an adequette mix of spike classes
def mix(list1,list2,list3,list4,list5):

    combidata = []
    combidata.extend(list1)
    combidata.extend(list2)
    combidata.extend(list3)
    combidata.extend(list4)
    combidata.extend(list5)
    shuffle(combidata) #shuffle items in list to closer resemble real data set 
    return combidata

# function to create training, validation and test data subsets 
# from inputted lists of classes and indexes
def CreateDataSets(Class,Index):


    #Set up variables to store data into different peak classes and 
    #respective indexes 
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    index1 = []
    index2 = []
    index3 = []
    index4 = []
    index5 = []

    #Sort Index and Class data by class of peak
    for type, loc in zip(Class, Index):
        
        if type == 1:
            class1.append(type)
            index1.append(loc)
        if type == 2:
            class2.append(type)
            index2.append(loc)
        if type == 3:
            class3.append(type)
            index3.append(loc)
        if type == 4:
            class4.append(type)
            index4.append(loc)
        if type == 5:
            class5.append(type)
            index5.append(loc)

    #Split data for each class of peak into train (70%), validate (15%) and test(15%) sets
    train1, validate1, test1 = split(class1,index1)
    train2, validate2, test2 = split(class2,index2)
    train3, validate3, test3 = split(class3,index3)
    train4, validate4, test4 = split(class4,index4)
    train5, validate5, test5 = split(class5,index5)

    #Merge data from different peak classes to prodcue to
    #esnure each type of set has an adequette mix of peak classes
    train_data = mix(train1,train2,train3,train4,train5)
    vali_data = mix(validate1,validate2,validate3,validate4,validate5)
    test_data = mix(test1,test2,test3,test4,test5)

    return train_data,vali_data,test_data


# function to make CSV file to store data 
def makeCSV(setdata,bigdata,type):
    #change file name depending on data type
    if type ==1:
        f_name = "train_data.csv"
    if type ==2:
        f_name = "validate_data.csv"
    if type ==3:
         f_name = "test_data.csv"

    # create CSV file and start writing onto it
    f = open(f_name,"w")
    writer = csv.writer(f)

    #iterate through data list
    for item in setdata:
        row = [] #create a row for each detected peak
        row.append(item[0]) # append the class to the start of the row  
        lowbound = item[1]-25 # determine the start of the spike
        upperbound = item[1]+40 # determine the end of the spike
        range = bigdata[lowbound:upperbound] #extract spike data from dataset
        row.extend(range) #append spike data onto row 
        writer.writerow(row) #add row to the CSV file
    f.close
    return

# function to read selected CSV file, extract data held in it 
# and process the data so that it in a suitable form to be input
# into the MLP
def get_inputs_targets(type):
    
    #open specific CSV file depedning on type selected
    if type ==1:
       data_file = open("train_data.csv", 'r')
    if type ==2:
        data_file = open("validate_data.csv", 'r')
    if type ==3:
         data_file = open("test_data.csv", 'r')

    # read data in CSV file
    data_list = data_file.readlines()
    data_file.close()

    #define MLP inputs
    output_nodes = 5
    inputs = []
    targets = []

    for data in data_list:
        all_values = data.split(',')# split data in list by the commas
        input_row = np.asfarray(all_values[1:]) # convert row data into an array except for class value
        inputs.append(input_row) #add array to MLP input array
        # create the target output values (all 0.01, except the desired label which is 0.99)
        target_row = np.zeros(output_nodes) + 0.01
        target_row[int(all_values[0])-1] = 0.99 # set the class label for this record
        targets.append(target_row) #add array to MLP target input

    targets = np.array(targets) # transform target list into an array
    inputs = normaliser(inputs) # normalise all spike data 

    return inputs, targets


# function to match the detected spike peaks with 
# their associated class by comparing detected locations 
# against provided spike peak location and class data 
def MatchClassIndex(PredIndex, Index, Class):

    combilists = zip(Index,Class) #combine provided index and class lists 
    sortedlists = sorted(combilists) #sort the lists in ascending order of location within data set
    tuples = zip(*sortedlists) #create tuple from ordered list

    ti, tc = [list(t) for t in tuples]

    #create varibales to store matched spike indexes and classes
    specific = []
    solutions = []

    #create "not found" variable to determine how 
    # many detected peaks have not been matched 
    nf = 0 

    #create list to determine wehther a given spike index
    #has already been matched to
    alreadychosen = [(0)for i in tc] 
    
    #iterate through detected spike locations
    for i in PredIndex:

        #iterate through reference spkike locations
        for g in range(0,len(ti)):
            # check whether detected spike and reference spike are within
            # a suitable distance of each other to be the same spike
            if i - ti[g] <= 20: 
                if alreadychosen[g] == 0: #check whether reference spike has been matched already
                    #match detcted spike location with reference 
                    # class of matched reference spike location
                    specific.append(i)
                    specific.append(tc[g])
                    solutions.append(specific)
                    specific = []
                    alreadychosen[g] = 1 #label this reference spike location as matched 
                    break
        #check whether a detected spike has been matched with a ref. spike
        if g == len(ti)-1 and i != PredIndex[len(PredIndex)-1]:
            nf = nf+1 

    #check how many reference spikes were not matched
    unmacthed = len(ti)-sum(alreadychosen) 
    DetIndex = []
    DetClass = []

    # create new index and class lists with 
    #detcted spike locations and associated classes
    for specific in solutions:
        DetIndex.append(specific[0])
        DetClass.append(specific[1])
    
    return DetIndex, DetClass

#function to make CSV file for submission data
def makeSubCSV(setdata,bigdata):

    setdata = setdata[1:] #discard inital noise spike to prevent errors
    f_name = "submission_data.csv" 
    f = open(f_name,"w") #open submission CSV file
    writer = csv.writer(f)

    #create a row for each detected peak
    for item in setdata:
        row = [] #create row for spike data
        lowbound = item-25 # determine the start of the spike
        upperbound = item+40 # determine the end of the spike
        range = bigdata[lowbound:upperbound] #extract define spike from submission data
        row.extend(range) #add extracted spike data to CSV file
        writer.writerow(row)
    f.close
    return

# function to read submission CSV file, extract data held in it 
# and process the data so that it in a suitable form to be input
# into the MLP
def getinputs():
    data_file = open("submission_data.csv", 'r') #open CSV file
    data_list = data_file.readlines() #read CSV file
    data_file.close()

    inputs = [] #create input array list for MLP

    for data in data_list:
        all_values = data.split(',')# split data in list by the commas
        input_row = np.asfarray(all_values[:])# convert row data into an array 
        inputs.append(input_row) #append row to MLP input list

    inputs = normaliser(inputs) #normalise input data 

    return inputs



