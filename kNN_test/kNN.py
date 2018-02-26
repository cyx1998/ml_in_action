import matplotlib.pyplot as plt
import numpy as np
import operator
from os import listdir

#test data
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)#Return the sum of the array elements over the given axis
    sortedDistIndicies = sqDistances.argsort()#Returns the indices that would sort this array
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]
    
def file2matrix(filename):
    loveDictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[:3]
        classLabelVector.append(loveDictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

#plot data
def file2figure(datingDataMat, datingLabels):
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    ax1.scatter(datingDataMat[:,0], datingDataMat[:,1],
               s=15*np.array(datingLabels),
               c=np.array(datingLabels))
    ax1.set_xlabel('Flight')
    ax1.set_ylabel('Games')
    ax2.scatter(datingDataMat[:,0], datingDataMat[:,2],
               s=15*np.array(datingLabels),
               c=np.array(datingLabels))
    ax2.set_xlabel('Flight')
    ax2.set_ylabel('Ice-cream')
    ax3.scatter(datingDataMat[:,1], datingDataMat[:,2],
               s=15*np.array(datingLabels),
               c=np.array(datingLabels))
    ax3.set_xlabel('Games')
    ax3.set_ylabel('Ice-cream')
    plt.show()

#normalization
def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def datingClassTest(k=3):
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m], k)
        print("the classifier came back with: %d, the real answer is %d"\
              % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("the total error rate is: %f" % float(errorCount/numTestVecs))

def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat,
                                 datingLabels, k=3)
    print("You will probably like this person:",
          resultList[classifierResult - 1])

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat,
                                     hwLabels, k=3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1
    print("\nthe total error rate is: %f" % float(errorCount/mTest))
        
