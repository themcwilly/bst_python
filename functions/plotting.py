import matplotlib.pyplot as plt
import pandas as pd

#Plot the data
def plot(df, options = {'title': 'Title', 'xaxis':'x-axis', 'yaxis':'y-axis'}):
    ax = df.plot(figsize=(15, 8), title=options['title'])
    ax.set_xlabel(options['xaxis'])
    ax.set_ylabel(options['yaxis'])
    plt.show()

def plot_hist(df, options = {'stacked': True, 'bins': 50, 'orientation': 'vertical'}):
    df.plot(kind='hist', stacked=options['stacked'], bins=options['bins'], orientation=options['orientation'])
    plt.show()

def plot_bw(df, options = {'vert': True}):
    color = {'boxes': 'DarkGreen', 'whiskers': 'r'}
    df.plot(kind='box', figsize=(10, 5), color=color, vert=options['vert'])
    plt.show()

def plot_scatter_matrix(df):
    pd.plotting.scatter_matrix(df, figsize=(8, 8), diagonal='kde', color='r')
    plt.show()

def plot_scatter(x_data, y_data, colors = ['red']):
    plt.scatter(x_data, y_data, c=colors)
    plt.show()