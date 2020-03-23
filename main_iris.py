import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functions.clustering import Cluster

import functions.plotting as plot

if __name__ == '__main__':
    # download a flower dataset
    iris = sns.load_dataset('iris')
    print(iris.head())

    #Plot the data
    plot.plot(iris)
    plot.plot_hist(iris)
    plot.plot_bw(iris)
    plot.plot_scatter_matrix(iris)

    # Do a little machine learning
    # k-means clustering on the 3 different types of flowers
    # change the dataframe so that text data is not present
    iris_df = iris.drop(['species'],axis=1)
    x_axis = iris_df['sepal_length'].values  # Sepal Length
    y_axis = iris_df['sepal_width'].values  # Sepal Width
    cluster = Cluster(df=iris_df)
    cluster.Model(n_clusters=3)
    predictions = cluster.Predict(iris_df)

    plot.plot_scatter(x_axis, y_axis, colors=predictions)

