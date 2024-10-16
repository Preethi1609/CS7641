
'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np


class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):  # No need to implement
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters


    def init_centers(self):
            """
            Initialize the centers randomly.

            Returns:
                self.centers : K x D numpy array, the centers.
            """
            unique_points = np.unique(self.points, axis=0)
            num_unique_points = unique_points.shape[0]
            
            if num_unique_points < self.K:
                raise ValueError("Number of unique points is less than the number of clusters (k).")

            shuffled_indices = np.random.permutation(num_unique_points)
            initial_centers_indices = shuffled_indices[:self.K]
            self.centers = unique_points[initial_centers_indices]
            return self.centers

    def kmpp_init(self):
        """
        Use k-means++ initialization to select initial cluster centers.

        Return:
            self.centers : K x D numpy array, the centers.
        """
        sample_size = max(len(self.points) // 100, 1)
        indices = np.random.choice(len(self.points), size=sample_size, replace=False)
        pts = self.points[indices]
        self.centers = np.array([pts[np.random.choice(sample_size)]])
        for _ in range(1, self.K):
            distances = np.array([min(np.linalg.norm(point - center) ** 2 for center in self.centers) for point in pts])
            next_center_index = np.argmax(distances)
            next_center = pts[next_center_index]
            self.centers = np.vstack((self.centers, next_center))
        return self.centers


    def update_assignment(self):  # [5 pts]
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison. 
        """       
        distances = pairwise_dist(self.points, self.centers)
        self.assignments = np.argmin(distances, axis=1)
        return self.assignments 



    def update_centers(self):  # [5 pts]
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """
        self.centers = np.array([self.points[self.assignments == k].mean(axis=0) for k in range(self.K)])
        return self.centers


    def get_loss(self):  # [5 pts]
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        distances = pairwise_dist(self.points, self.centers)
        squared_distances = distances[np.arange(len(distances)), self.assignments] ** 2
        self.loss = np.sum(squared_distances)
        return self.loss

    def train(self):    # [10 pts]
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned, 
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference 
                     in loss compared to the previous iteration is less than the given 
                     relative tolerance threshold (self.rel_tol). 
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.   
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """
        for iteration in range(self.max_iters):
            assign = self.update_assignment()
            # print("assignments: ", assign)

            cens = self.update_centers()
            # print("update centers: ", cens)

            empty_clusters = np.where(np.bincount(self.assignments, minlength=self.K) == 0)[0]
            if len(empty_clusters) > 0:
                for empty_cluster in empty_clusters:
                    random_index = np.random.choice(len(self.points))
                    self.centers[empty_cluster] = self.points[random_index]
                    self.assignments[random_index] = empty_cluster
            prev_loss = self.loss
            self.get_loss()
            if iteration > 0 and (prev_loss - self.loss) / prev_loss < self.rel_tol:
                break

        return self.centers, self.assignments, self.loss


def pairwise_dist(x, y):  # [5 pts]
        np.random.seed(1)
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """
        x_reshaped = x
        y_reshaped = y
        x_squared = np.sum(x_reshaped**2, axis=1, keepdims=True)
        y_squared = np.sum(y_reshaped**2, axis=1, keepdims=True)
        xy_dot = np.dot(x_reshaped, y_reshaped.T)
        
        dist_squared = x_squared - 2 * xy_dot + y_squared.T
        return np.sqrt(np.maximum(dist_squared, 0))




def rand_statistic(xGroundTruth, xPredicted): # [5 pts]
    """
    Args:
        xPredicted : N x 1 numpy array, N = no. of test samples
        xGroundTruth: N x 1 numpy array, N = no. of test samples
    Return:
        Rand statistic value: final coefficient value as a float
    """
    N = len(xGroundTruth)
    TP = TN = FP = FN = 0
    
    for i in range(N):
        for j in range(i+1, N):
            same_ground_truth = (xGroundTruth[i] == xGroundTruth[j])
            same_predicted = (xPredicted[i] == xPredicted[j])
            
            if same_ground_truth and same_predicted:
                TP += 1
            elif not same_ground_truth and not same_predicted:
                TN += 1
            elif same_ground_truth and not same_predicted:
                FN += 1
            elif not same_ground_truth and same_predicted:
                FP += 1

    rand_index = (TP + TN) / (TP + FP + FN + TN)
    return rand_index

