import numpy as np
from kmeans import pairwise_dist
class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        n = len(self.dataset)
        cluster_idx = np.full(n, -1) 
        visitedIndices = set()
        C = 0
        for i in range(n):
            if i not in visitedIndices:
                neighborIndices = self.regionQuery(i)

                if len(neighborIndices) < self.minPts:
                    cluster_idx[i] = -1
                else:
                    self.expandCluster(i, neighborIndices, C, cluster_idx, visitedIndices)
                    C+=1

        return cluster_idx


    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints:  
            1. np.concatenate(), and np.sort() may be helpful here. A while loop may be better than a for loop.
            2. Use, np.unique(), np.take() to ensure that you don't re-explore the same Indices. This way we avoid redundancy.
        """
        queue = list(neighborIndices)

        while queue:
            point = queue.pop(0)
            
            if point not in visitedIndices:
                visitedIndices.add(point)
                new_neighbors = self.regionQuery(point)
                
                if len(new_neighbors) >= self.minPts:
                    new_neighbors.sort()
                    queue.extend(np.setdiff1d(new_neighbors, list(visitedIndices)))
                
            cluster_idx[point] = C

        
    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        reshaped_pointIndex = self.dataset[pointIndex]
        reshaped_pointIndex = reshaped_pointIndex[np.newaxis, :]
        distances = pairwise_dist(reshaped_pointIndex, self.dataset)
        indices = np.argwhere(distances <= self.eps)[:, -1]
        return indices
