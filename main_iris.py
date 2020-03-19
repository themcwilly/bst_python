import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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


    # Do a little machine learning
    # k-means clustering on the 3 different types of flowers
    # change the dataframe so that text data is not present

    iris_df = iris.drop(['species'],axis=1)
    model = KMeans(n_clusters=3)
    model.fit(iris_df)

    # Predicitng a single input
    predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])
    # Prediction on the entire data
    all_predictions = model.predict(iris_df)
    # Printing Predictions
    print(predicted_label)
    print(all_predictions)
    # turn values back into target data
    target = iris['species'].unique()

    #assign target to values
    label = {0: 'red', 1: 'blue', 2: 'green'}

    # Dataset Slicing
    x_axis = iris_df['sepal_length'].values  # Sepal Length
    y_axis = iris_df['sepal_width'].values  # Sepal Width

    # Plotting
    plt.scatter(x_axis, y_axis, c=all_predictions)
    plt.show()

