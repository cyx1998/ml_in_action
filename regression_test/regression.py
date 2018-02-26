import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataArr, labelArr = [], []
    with open(fileName) as fr:
        numFeat = len(fr.readline().split('\t')) - 1
        fr.seek(0,0) #reset the handle
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            dataArr.append(lineArr)
            labelArr.append(float(curLine[-1]))
    return dataArr, labelArr

#wHat = (X^T*X)^-1*X^T*y
def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    ws = xTx.I * xMat.T * yMat
    return ws
    
        
def standRegresFigure(xArr, yArr, ws):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0]) #Masked arrays must be 1-Dim
    xCopy = xMat.copy()
    xCopy.sort(axis=0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1], yHat)
    yHatO = xMat * ws
    print("the Correlation Coefficient is: ", np.corrcoef(yHatO.T, yMat)) #
    plt.show()

#Locally Weighted Linear Regression
#wHat = (X^T*W*X)^-1*X^T*W*y
def locWeightedRegres(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = np.exp(diffMat * diffMat.T/(-2 * k**2))
    xTwx = xMat.T * weights * xMat
    if np.linalg.det(xTwx) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    ws = xTwx.I * xMat.T * weights * yMat
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros((m))
    for i in range(m):
        yHat[i] = locWeightedRegres(testArr[i], xArr, yArr, k)
    return yHat

def lwlrFigure(testArr, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yHat = lwlrTest(testArr, xArr, yArr, k)
    srtInd = xMat[:,1].argsort(axis=0)
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], yArr, c='r')
    plt.show()

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

#wHat = (X^T*X+λI)^-1*X^T*y
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    ws = denom.I * xMat.T * yMat
    return ws

def regularize(xMat):
    xVar = np.var(xMat, axis=0) #variance of x
    xMean = np.mean(xMat, axis=0)
    return (xMat - xMean)/xVar

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, axis=0) #mean value of y
    yMat = yMat - yMean #regularize
    xMat = regularize(xMat) #regularize
    numTestPts = 30
    wMat = np.mat(np.zeros((numTestPts, np.shape(xMat)[1])))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i,:] = ws.T
    return wMat


    
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, axis=0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssErr = rssError(yMat.A, yTest.A)
                if rssErr < lowestError:
                    lowestError = rssErr
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat
        
