from numpy import *


def loadDataSet(fileName):
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))
            dataSet.append(fltLine)
    return dataSet

def distEclud2(vecA, vecB):
    return sum(power(vecA - vecB, 2))

def randCent(dataMat, k):
    n = shape(dataMat)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataMat[:,j])
        rangeJ = float(max(dataMat[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return centroids

def kMeans(dataMat, k, distMeas=distEclud2, createCent=randCent):
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataMat, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist, minIndex = inf, -1
            for j in range(k):
                dist = distMeas(centroids[j,:], dataMat[i,:])
                if dist < minDist:
                    minDist, minIndex = dist, j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist
        print(centroids)
        for cent in range(k):
            ptsInclust = dataMat[nonzero(clusterAssment[:,0] == cent)[0]]
            centroids[cent,:] = mean(ptsInclust, axis=0)
    return centroids, clusterAssment
            
def biKMeans(dataMat, k, distMeas=distEclud2):
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataMat, axis=0)
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(centroid0, dataMat[j,:])
    while len(centList) < k:
        minSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataMat[nonzero(clusterAssment[:,0] == i)[0],:]
            centroidMat, splitClustAssment = kMeans(ptsInCurrCluster,
                                                    2, distMeas)
            SSESplit = sum(splitClustAssment[:,1])
            SSENotSplit = \
                sum(clusterAssment[nonzero(clusterAssment[:,0] != i)[0],1])
            print("SSESplit, SSENotSplit: ", SSESplit, SSENotSplit)
            if (SSESplit + SSENotSplit) < minSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAssment = splitClustAssment.copy()
                minSSE = SSESplit + SSENotSplit
        bestClustAssment[nonzero(bestClustAssment[:,0] == 1)[0],0] = \
            len(centList)
        bestClustAssment[nonzero(bestClustAssment[:,0] == 0)[0],0] = \
            bestCentToSplit
        print("the bestCentToSplit is: ", bestCentToSplit)
        print("the len of bestClustAssment is: ", len(bestClustAssment))
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0] == bestCentToSplit)[0],:] = \
            bestClustAssment
    return centList, clusterAssment  

