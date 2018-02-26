import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():
    dataMat = np.mat([[1.0, 2.1],
                      [2.0, 1.1],
                      [1.3, 1.0],
                      [1.0, 1.0],
                      [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    retArr = np.ones((np.shape(dataMat)[0], 1))
    if threshIneq == 'lt':
        retArr[dataMat[:,dimen] <= threshVal] = -1.0
    else:
        retArr[dataMat[:,dimen] > threshVal] = -1.0
    return retArr
 
def buildStump(dataArr, classLabels, D):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMat[:,i].min()
        rangeMax = dataMat[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = rangeMin + float(j) * stepSize
                predictedVals = stumpClassify(dataMat,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst
                
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr,classLabels,D)
        #print('D:', D.T)
        alpha = float(0.5 * np.log((1 - error)/max(error, 1e-16))) #interprete matrix to float
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print('classEst:', classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst) #1 for incorrect samples, -1 for correct samples
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha * classEst
        #print('aggClassEst:', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,
                                np.ones((m,1)))
        errorRate = aggErrors.sum()/m #built-in sum(matrix or array) will return a matrix or array; numpy.sum(matrix or array) or ndarray.sum() will return a float
        #print("total error:", errorRate, '\n')
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

def adaClassify(dataToClass, classifierArr):
    dataMat = np.mat(dataToClass)
    m = np.shape(dataMat)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)        

def loadDataSet(fileName):
    dataArr, labelArr = [], []
    with open(fileName) as fr:
        numFeat = len(fr.readline().split('\t')) 
        fr.seek(0,0) #reset the handle
        for line in fr.readlines():
            lineArr = []
            currline = line.strip().split('\t')
            for i in range(numFeat-1):
                lineArr.append(float(currline[i]))
            dataArr.append(lineArr)
            labelArr.append(float(currline[-1]))
    return dataArr, labelArr

#ROC(receiver operating characteristic) curve
#x = FP/(FP + TN) = FP/N
#y = TP/(TP + FN) = TP/P
def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClass = np.sum(np.array(classLabels) == 1.0)
    yStep = 1/float(numPosClass)
    xStep = 1/float(len(classLabels) - numPosClass)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX, delY = 0, yStep
        else:
            delX, delY = xStep, 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0,1], [0,1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)
