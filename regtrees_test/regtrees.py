import numpy as np

def loadDataSet(fileName):
    dataArr = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine)) #
            dataArr.append(fltLine)
    return dataArr

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0, mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

#createTree(mat, func, func, tuple)
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spFeat'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS, tolN = ops[0], ops[1] #tolS: admissible decline in error; tolN: minimum sample number of the split
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = np.shape(dataSet)
    S = errType(dataSet) #entropy
    bestS, bestFeat, bestValue = np.inf, 0, 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN: continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestFeat = featIndex
                bestValue = splitVal
                bestS = newS
    if S - bestS  < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestFeat, bestValue)
    if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
        return None, leafType(dataSet)
    return bestFeat, bestValue
            
def isTree(obj):
    return isinstance(obj, dict)

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0

def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    lSet, rSet = binSplitDataSet(testData, tree['spFeat'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'],2)) + \
                       np.sum(np.power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = np.sum(np.power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print('merge')
            return treeMean
    return tree

def linearSolve(dataSet):
    m,n = np.shape(dataSet)
    xMat = np.mat(np.ones((m,n)))
    yMat = np.mat(np.ones((m,1)))
    xMat[:,1:n] = dataSet[:,:n-1]
    yMat = dataSet[:,-1]
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        raise NameError('''This matrix is singular, cannot do inverse,
                        try increasing the second value of ops''')
    ws = xTx.I * xMat.T * yMat
    return ws, xMat, yMat

def modelLeaf(dataSet):
    ws, xMat, yMat = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, xMat, yMat = linearSolve(dataSet)
    yHat = xMat * ws
    return np.sum(np.power(yHat - yMat,2))

def regTreeEval(model, inData):
    return float(model)

def modelTreeEval(model, inData):
    n = np.shape(inData)[1]
    xMat = np.mat(np.ones((1, n+1)))
    xMat[:,1:n+1] = inData
    return float(xMat * model)

def treeForecast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spFeat']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForecast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForecast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
        
def createForecast(tree, testData, modelEval=regTreeEval):
    m = np.shape(testData)[0]
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForecast(tree, testData[i,:], modelEval)
    return yHat
        
