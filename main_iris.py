import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # download a flower dataset
    iris = sns.load_dataset('iris')
    print(iris.head())

    #Plot the data
    ax = iris.plot(figsize=(15, 8), title='Iris Dataset')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    plt.show()

    #make a histogram
    iris.plot.hist()
    plt.show()
    iris.plot(kind='hist', stacked=True, bins=50)
    plt.show()
    iris.plot(kind='hist', stacked=True, bins=50, orientation='horizontal')
    plt.show()

    #box and whisker
    color = {'boxes': 'DarkGreen', 'whiskers': 'r'}
    iris.plot(kind='box', figsize=(10, 5), color=color)
    plt.show()
    iris.plot(kind = 'box', figsize=(10,5), color = color, vert = False)
    plt.show()

    #scatter
    pd.plotting.scatter_matrix(iris, figsize=(8, 8), diagonal='kde', color='r')
    plt.show()



