import numpy as np

class KMeansClustering:
    def __init__(self, k=3, max_iter=200, threshold=0.0001):
        self.k = k
        self.max_iter = max_iter
        self.threshold = threshold
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def initialize_centroids(self, X):
        return np.random.uniform(
            np.amin(X, axis=0),
            np.amax(X, axis=0),
            size=(self.k, X.shape[1])
        )

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iter):
            y = []
            for data_point in X:
                distances = self.euclidean_distance(data_point, self.centroids)
                y.append(np.argmin(distances))

            y = np.array(y)

            cluster_centers = []
            for i in range(self.k):
                cluster_points = X[y == i]
                if len(cluster_points) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(cluster_points, axis=0))

            cluster_centers = np.array(cluster_centers)

            if np.max(np.abs(self.centroids - cluster_centers)) < self.threshold:
                break

            self.centroids = cluster_centers

        return y

    def predict(self, X):
        labels = []
        for data_point in X:
            distances = self.euclidean_distance(data_point, self.centroids)
            labels.append(np.argmin(distances))
        return np.array(labels)

