import random
import matplotlib.pyplot as plt
import numpy as np
import warnings

def loadDataSet():
    dataArr, labelArr = [], []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataArr.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelArr.append(int(lineArr[2]))
    return dataArr, labelArr

def sigmoid(inX):
    return  1/(1 + np.exp(-inX))
    
def gradAscent(dataArr, labelArr):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr)
    m, n = dataMat.shape
    alpha = 0.001
    maxCycles = 500
    weightArr = np.ones((1, n))
    for k in range(maxCycles):
        hMat = sigmoid(dataMat * weightArr.T)
        errorMat = labelMat - hMat.T
        weightArr = weightArr + alpha * errorMat * dataMat #here type of weightArr is interpreted to matrix
    return weightArr.getA().ravel()

def stocGradAscent0(dataArr, labelArr):
    m, n = np.shape(dataArr)
    alpha = 0.01
    weightArr = np.ones(n)
    for i in range(m):
        h = sigmoid(np.inner(dataArr[i], weightArr))
        error = labelArr[i] - h #numpy broad
        weightArr = weightArr + alpha * error * dataMat[i]
    return weightArr

def stocGradAscent1(dataArr, labelArr, numIter=25):
    m, n = dataArr.shape
    weightArr = np.ones(n)
    for i in range(numIter):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4/(1+j+i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(np.inner(dataArr[randIndex], weightArr)) #it's much faster than using matrix product to calculate inner product of 2 vectors
            error = labelArr[randIndex] - h
            weightArr = weightArr + alpha * error * dataArr[randIndex]
            del(dataIndex[randIndex])
    return weightArr
    
def plotBestFit(weights):
    dataArr, labelArr = loadDataSet()
    dataMat = np.mat(dataArr)
    m = len(labelArr)
    xcord1, ycord1 = [], []
    xcord0, ycord0 = [], []
    for i in range(m):
        if int(labelArr[i]) == 1:
            xcord1.append(dataMat[i,1])
            ycord1.append(dataMat[i,2])
        else:
            xcord0.append(dataMat[i,1])
            ycord0.append(dataMat[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord0, ycord0, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def classifyMat(dataMat, weightMat):
    probMat = sigmoid(dataMat * weightMat.T)
    index1 = probMat > 0.5
    index0 = probMat <= 0.5
    probMat[index1] = 1
    probMat[index0] = 0
    return probMat.T
        
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet, trainingLabels = [], []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels,
                                   numIter=1000)
    testSet, testLabels = [], []
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[21]))
    probMat = classifyMat(np.mat(testSet), np.mat(trainWeights))
    numTest = len(testLabels)
    errorCount = np.shape(probMat[probMat != testLabels])[1]
    errorRate = errorCount/numTest
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests, errorSum = 10, 0.0
    for i in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f"
          % (numTests, errorSum/numTests))
    
