from sklearn.cluster import KMeans
import functions.plotting as plot

class Cluster:
    model = None

    def __init__(self, df = None):
        self.df = df

    def Model(self, n_clusters=3):
        if self.df is None:
            print('Please initialize with valid dataframe.')
            return
        self.model = KMeans(n_clusters=n_clusters)
        self.model.fit(self.df)
        # model.predict([[7.2, 3.5, 0.8, 1.6]])

    def Predict(self, data):
        if self.model is None:
            print('There hasn\'t been a trained model yet.')
            return
        return self.model.predict(data)
