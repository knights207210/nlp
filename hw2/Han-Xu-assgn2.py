
# coding: utf-8



#Feature Extraction


import numpy as np
import csv
import math
import string



class featureExtraction:
    def __init__(self, name):
        self.fileName = name
        self.review = []
        self.ID = []
        self.label = []
        self.posWords = []
        self.negWords = []
        self.features = []
        self.ID_Review = []
    
    def readReview(self):
        with open(self.fileName, 'r') as fp:
            for i,line in enumerate(fp):
                #split ID and review
                self.ID_Review = line.split('\t')
                self.ID.append(self.ID_Review[0])
                self.review.append(self.ID_Review[1].rstrip('\n')) 
        if self.fileName == "hotelPosT-train.txt":
            self.label = [1]*len(self.review)
        elif self.fileName == "hotelNegT-train.txt":
            self.label = [0]*len(self.review)
        else:
            self.label = [-1]*len(self.review)
        
        return self.ID, self.review, self.label
                
    def readNegPosWords(self):
        with open("positive-words.txt", 'r') as fp:
            self.posWords=[]
            for i,line in enumerate(fp):
                self.posWords.append(line.rstrip('\n'))
        with open("negative-words.txt", 'r') as fp:
            self.negWords=[]
            for i,line in enumerate(fp):
                self.negWords.append(line.rstrip('\n'))
                
    def extractFeatures(self,review):
        
        for r in review:
            wordList = r.split()
            pronouns = ["i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]
            
            #feature 6:
            logWordCount = math.log(len(wordList))
        
            posCount = 0
            negCount = 0
            noInOrNot = 0
            excMarkInOrNot = 0
            pronounsCount = 0
            
            for word in wordList:
                #feature 5:
                if '!' in word:
                    excMarkInOrNot = 1
                    
                #feature 4:
                if (word.lower() in pronouns) or ('\'' in word and word.split('\'')[0].lower() in pronouns):
                    pronounsCount += 1
                    
                #remove punctuation
                punc = string.punctuation
                punc = punc.replace('-','') # for some special cases like "kid-friendly"
                word = word.translate(str.maketrans('', '', punc))
        
                #feature 1:
                if word.lower() in self.posWords:
                    posCount += 1
                
                #feature 2:
                if word.lower() in self.negWords:
                    negCount += 1
                    
                #feature 3:
                if word.lower()=='no':
                    noInOrNot = 1
                    
            self.features.append([posCount, negCount, noInOrNot, pronounsCount, excMarkInOrNot, logWordCount])
            
        return self.features

def writeToCSV(ID,features,label,name):
    #arrange format: ID+feature+label
    writeToCSV_format = []
    for i in range(len(ID)):
        writeToCSV_format.append([ID[i]]+features[i]+[label[i]])
    with open(name,'w') as f:
        writer = csv.writer(f)
        writer.writerows(writeToCSV_format)
            
def writeToCSV_allData():  #write all provided ID+feature+label to feature.csv
    posFeature = featureExtraction("hotelPosT-train.txt")
    posID, posReview, posLabel = posFeature.readReview()
    posFeature.readNegPosWords()
    posFeatures = posFeature.extractFeatures(posReview)

    negFeature = featureExtraction("hotelNegT-train.txt")
    negID, negReview, negLabel = negFeature.readReview()
    negFeature.readNegPosWords()
    negFeatures = negFeature.extractFeatures(negReview)

    #write to features to feature.csv
    writeToCSV(posID+negID, posFeatures+negFeatures, posLabel+negLabel, 'feature.csv')



# logistic regression model 


class LogReg:
    def __init__(self, train_set, test_set, eta=0.1):
        
        self.train_set = train_set
        self.test_set = test_set 
        
        self.w = np.zeros_like(train_set[0].x)

        self.eta = eta
        
        self.train_acc = []
        self.test_acc = []  
        self.train_nll = []
        self.test_nll = []
        
    def sigmoid(self,x):

        return 1.0 / (1.0 + np.exp(-x)) 

    def compute_progress(self, examples):
        #loss
        NLL = 0.0
        #accuracy
        num_correct = 0
        for ex in examples:
            # compute prob prediction
            p = self.sigmoid(self.w.dot(ex.x))
            # update negative log likelihood
            NLL = NLL - np.log(p) if ex.y==1 else NLL - np.log(1.0-p)
            # update number correct 
            num_correct += 1 if np.floor(p+.5)==ex.y else 0

        return NLL, float(num_correct) / float(len(examples))
    
    def train(self, num_epochs=5, isVerbose=False, report_step=5):
    
        iteration = 0
        for pp in range(num_epochs):
            # shuffle the data  
            np.random.shuffle(self.train_set)
            # loop over each training example
            for ex in self.train_set:
                # perform SGD update of weights 
                self.sgd_update(ex)
                # record progress 
                if iteration % report_step == 1:
                    train_nll, train_acc = self.compute_progress(self.train_set)
                    self.train_nll.append(train_nll)
                    self.train_acc.append(train_acc)
                    if isVerbose:
                        print("Update {: 5d}  TrnLoss {: 8.3f}  TrnAcc {:.3f}"
                             .format(iteration-1, train_nll, train_acc))
                iteration += 1
    
    def sgd_update(self, train_example):
        
        sig_minus_y = self.sigmoid(self.w.dot(train_example.x))-train_example.y
        for k in np.nonzero(train_example.x)[0]:
            #unregularized part
            gradient = sig_minus_y*train_example.x[k]
            self.w[k] = self.w[k]-self.eta*gradient
            
    def test(self, threshold = 0.5):
        #accuracy
        num_correct = 0
        #loss
        NLL = 0.0
        
        #test set label list
        labelList = []
        
        
        # test set label is null:
        if self.test_set[0].y == -1:
            for ex in self.test_set:
            # compute prob prediction
                p = self.sigmoid(self.w.dot(ex.x))
                if p >= threshold:
                    labelList.append('POS')
                else:
                    labelList.append('NEG')
            return labelList
        
        #dev set
        else:
            for ex in self.test_set:
            # compute prob prediction
                p = self.sigmoid(self.w.dot(ex.x))
            # update negative log likelihood
                NLL = NLL - np.log(p) if ex.y==1 else NLL - np.log(1.0-p)
            # update number correct 
                num_correct += 1 if np.floor(p+.5)==ex.y else 0              
            print("dev Set Loss:" + str(NLL) +"  dev Set Accuracy: " +str(float(num_correct) / float(len(self.test_set))))       
            return []

            
            
    

#read trainset and devset

class dataset:
    def __init__(self, label, features):
        self.y = label
        self.x = features
        
def readFile(name):
    data = []
    with open(name, 'r') as fp:
        for i,line in enumerate(fp):
            arr = line.split(',')
            #print(dataset(arr[-1], arr[1:-1]).x)
            data.append(dataset(int(arr[-1].rstrip('\n')), list(np.array(arr[1:-1]).astype(np.float))))
            
    return data


#split train and test set
def getTrainTestSet(fileName):
    allData = readFile(fileName)
    np.random.shuffle(allData)
    trainSet = allData[0:int(0.8*len(allData))]
    devSet = allData[int(0.8*len(allData)):]
    
    return trainSet, devSet





#bias
def addBias(Set):
    for i in range(len(Set)):
        Set[i].x = Set[i].x+[1] 
    
    


#######################################################################################
#extract features from provided data
print("#######################################################################################")
print("extract features from provided data")
writeToCSV_allData()

print("features for assignment part 1 has been written to 'feature.csv'")

#trainSet and devSet
print("#######################################################################################")
print("split to trainSet and devSet")
trainSet, devSet = getTrainTestSet("feature.csv")

print("start training, learning_rate = 0.001")
addBias(trainSet)
addBias(devSet)
LR = LogReg(trainSet, devSet, 0.001)
LR.train(num_epochs=100, isVerbose=True, report_step = 200)

print("devSet result")
LR.test()

#use all data as trainSet and run testSet
print("#######################################################################################")
print("get all data as trainSet")
trainData = readFile('feature.csv')

print("extract features for testSet")
testFeature = featureExtraction("HW2-testset.txt")
testID, testReview, testLabel = testFeature.readReview()
testFeature.readNegPosWords()
testFeatures = testFeature.extractFeatures(testReview)
writeToCSV(testID, testFeatures, testLabel, 'testFeature.csv')
testSet = readFile('testFeature.csv')

print("start training, learning_rate = 0.001")
addBias(trainData)
addBias(testSet)
LR = LogReg(trainData, testSet, 0.001)
LR.train(num_epochs=100, isVerbose=True, report_step = 200)

print("write test result to txt")
preLabel = LR.test()
with open('Han-Xu-assgn2-out.txt','w') as f:
     for i in range(len(testID)):
         f.write(testID[i]+'\t'+preLabel[i]+'\n')

