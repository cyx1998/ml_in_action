import trees
import treePlotter

if __name__ == "__main__":
    fr = open('lenses.txt')
    lensesData = [line.strip().split('\t') for line in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = trees.createTree(lensesData, lensesLabels)
    print(lensesTree)
    treePlotter.createPlot(lensesTree)
