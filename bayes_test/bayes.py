#-*- coding: utf-8 -*-

import operator
import random
import numpy as np
import re
import feedparser

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1 represents abusive words, 0 represents common words
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        #else:
            #print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/numTrainDocs
    p0Num, p1Num = np.ones(numWords), np.ones(numWords)
    p0Denom, p1Denom = 2.0, 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom) #ln(a*b) = ln(a) + ln(b) as mathematical process
    p0Vect = np.log(p0Num/p0Denom) #
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    listOfTokens = re.split('\\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList, classList, fullText = [], [], []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = set(range(50))
    testSet = set()
    while len(testSet) < 10:
        randIndex = random.randrange(len(trainingSet)-1)
        if randIndex in testSet: continue
        testSet.add(randIndex)
    trainingSet = trainingSet - testSet
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        classifyResult = classifyNB(np.array(wordVector), p0V, p1V, pSpam)
        if classifyResult != classList[docIndex]:
            errorCount += 1
    print("the error rate is:", errorCount/len(testSet))

def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1),
                        reverse=True)    
    return sortedFreq[:30]

#this func can predict where the docs come from
def localWords(feed1, feed0):
    docList, classList, fullText = [], [], []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary']) #visit RSS source
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0]) #remove the most frequent words
    trainingSet = set(range(2*minLen))
    testSet = set()
    while len(testSet) < 20:
        randIndex = random.randrange(len(trainingSet)-1)
        if randIndex in testSet: continue
        testSet.add(randIndex)
    trainingSet = trainingSet - testSet
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, p1Feed = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        classifyResult = classifyNB(wordVector, p0V, p1V, p1Feed)
        if classifyResult != classList[docIndex]:
            errorCount += 1
    print("the error rate is:", errorCount/len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY, topSF = [], []
    for i in range(len(p0V)):
        if p0V[i] > -5.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -5.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**"*12)
    for i, item in enumerate(sortedSF):
        print('rank %d: %s' % (i+1, item[0]))
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**"*12)
    for i, item in enumerate(sortedNY):
        print('rank %d: %s' % (i+1, item[0]))
