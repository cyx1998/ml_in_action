import matplotlib.pyplot as plt

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


#test data
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': \
                    {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': \
                    {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    matplotlib.axes.Axes.annotate(*args, **kwargs)
    
    Parameters:
        s : str
        
            The text of the annotation

            
        xy : iterable
        
            Length 2 sequence specifying the (x,y) point to annotate
            

        xytext : iterable, optional
        
            Length 2 sequence specifying the (x,y) to place the text at.
            
            If None, defaults to xy.

            
        xycoords : str, Artist, Transform, callable or tuple, optional

            The coordinate system that xy is given in.


        textcoords : str, Artist, Transform, callable or tuple, optional
        
            The coordinate system that xytext is given,which may be different
            than the coordinate system used for xy.


        arrowprops : dict, optional
        
            If not None, properties used to draw a FancyArrowPatch arrow between
            xy and xytext.
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt,
                            textcoords='axes fraction',
                            va='center', ha='center',
                            bbox=nodeType, arrowprops=arrow_args)

def plotMidText(centerPt, parentPt, txtString):
    xMid = (parentPt[0] - centerPt[0])/2 + centerPt[0]
    yMid = (parentPt[1] - centerPt[1])/2 + centerPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt=None):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    centerPt = (plotTree.xOff + (1 + numLeafs)/2 / plotTree.totalW,
                plotTree.yOff)
    plotMidText(centerPt, parentPt, nodeTxt)
    plotNode(firstStr, centerPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1/plotTree.totalD #forward
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], centerPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
                     centerPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), centerPt, str(key))
    plotTree.yOff = plotTree.yOff + 1/plotTree.totalD #regress

def createPlot(myTree):
    """
    matplotlib.pyplot.figure(num=None, figsize=None, dpi=None, facecolor=None,
    edgecolor=None, frameon=True, clear=False,
    FigureClass=<class 'matplotlib.figure.Figure'>, **kwargs)

    Parameters:
    
        num : integer or string, optional, default: none
        
            If not provided, a new figure will be created, and the figure number
            will be incremented. The figure objects holds this number in a
            number attribute.
            
            If num is provided, and a figure with this id already exists, make
            it active, and returns a reference to it.
            
            If this figure does not exists, create it and returns it.
            
            If num is a string, the window title will be set to this figure’s
            num.

        
        figsize : tuple of integers, optional, default: None
        
            width, height in inches.
            
            If not provided, defaults to rc figure.figsize.


        dpi : integer, optional, default: None

            resolution of the figure.
            
            If not provided, defaults to rc figure.dpi.


        facecolor :

            the background color.

            If not provided, defaults to rc figure.facecolor.

        edgecolor :

            the border color.

            If not provided, defaults to rc figure.edgecolor.


        frameon : bool, optional, default: True
        
            If False, suppress drawing the figure frame.


        FigureClass : class derived from matplotlib.figure.Figure

            Optionally use a custom Figure instance.


        clear : bool, optional, default: False

            If True and the figure already exists, then it is cleared.


        figure : Figure

            The Figure instance returned will also be passed to
            new_figure_manager in the backends, which allows to hook custom
            Figure classes into the pylab interface. Additional kwargs will be
            passed to the figure init function.
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf() #Clear the figure
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) #subplot(nrows, ncols, index, **kwargs)
    plotTree.totalW = getNumLeafs(myTree)
    plotTree.totalD = getTreeDepth(myTree)
    plotTree.xOff, plotTree.yOff = -0.5/plotTree.totalW, 1.0
    plotTree(myTree, (0.5, 1.0))
    plt.show()
    
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        maxDepth = max(maxDepth, thisDepth)
    return maxDepth


    
