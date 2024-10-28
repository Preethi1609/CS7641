import numpy as np
from typing import Tuple, List


class Regression(object):
    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:  # [5pts]
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """
        return np.sqrt(np.mean((pred - label) ** 2))

    def construct_polynomial_feats(
        self, x: np.ndarray, degree: int
    ) -> np.ndarray:  # [5pts]
        """
        Given a feature matrix x, create a new feature matrix
        which is all the possible combinations of polynomials of the features
        up to the provided degree

        Args:
            x:
                1-dimensional case: (N,) numpy array
                D-dimensional case: (N, D) numpy array
                Here, N is the number of instances and D is the dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat:
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
        Hints:
            - For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
            the bias term.
            - Example:
            For inputs x: (N = 3 x D = 2) and degree: 3,
            feat should be:

            [[[ 1.0        1.0]
                [ x_{1,1}    x_{1,2}]
                [ x_{1,1}^2  x_{1,2}^2]
                [ x_{1,1}^3  x_{1,2}^3]]

                [[ 1.0        1.0]
                [ x_{2,1}    x_{2,2}]
                [ x_{2,1}^2  x_{2,2}^2]
                [ x_{2,1}^3  x_{2,2}^3]]

                [[ 1.0        1.0]
                [ x_{3,1}    x_{3,2}]
                [ x_{3,1}^2  x_{3,2}^2]
                [ x_{3,1}^3  x_{3,2}^3]]]

        """
        if x.ndim == 1:
            N = len(x)
            feat = np.zeros((N, degree + 1))

            for d in range(degree + 1):
                feat[:, d] = x ** d

            return feat
        else:  
            N, D = x.shape
            feat = np.zeros((N, degree + 1, D))

            for d in range(degree + 1):
                feat[:, d, :] = x ** d

            return feat
        # raise NotImplementedError

    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:  # [5pts]
        """
        Using regression weights, predict the values for each data point in the xtest array

        Args:
            xtest: (N,1+D) numpy array, where N is the number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            weight: (1+D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        return np.dot(xtest, weight)
        # raise NotImplementedError

    # =================
    # LINEAR REGRESSION
    # =================

    def linear_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray
    ) -> np.ndarray:  # [5pts]
        """
        Fit a linear regression model using the closed form solution

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
        """
        return np.dot(np.linalg.pinv(xtrain), ytrain)

    def linear_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        D = xtrain.shape[1] - 1
        weight = np.zeros((1 + D, 1))
        loss_per_epoch = []
        N = len(ytrain)
        for _ in range(epochs):
            y_pred = xtrain @ weight
            error = ytrain - y_pred

            gradient = (xtrain.T @ error) / N
            weight += learning_rate * gradient
            rmse = self.rmse(xtrain @ weight, ytrain)
            loss_per_epoch.append(rmse)


        return weight, loss_per_epoch

    def linear_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            epochs: int, number of epochs
            learning_rate: float, value of regularization constant
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.


        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        D = xtrain.shape[1] - 1
        weight = np.zeros((1 + D, 1))
        loss_per_step = []

        N = len(ytrain)

        for _ in range(epochs):
            for i in range(N):
                xi = xtrain[i:i+1, :]
                yi = ytrain[i:i+1]
                # print("xi shape: ", xi.shape)
                # print("xi.T shape: ", xi.T.shape)
                y_pred = xi @ weight
                error = yi - y_pred
                # print("error shape: ", error.shape)
                # print("weight shape: ", weight.shape)
                gradient = (xi.T @ error)
                weight += learning_rate * gradient
                rmse = self.rmse(xtrain @ weight, ytrain)
                loss_per_step.append(rmse)

        return weight, loss_per_step
        # raise NotImplementedError

    # =================
    # RIDGE REGRESSION
    # =================

    def ridge_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float
    ) -> np.ndarray:  # [5pts]
        """
        Fit a ridge regression model using the closed form solution

        Args:
            xtrain: (N,1+D) numpy array, where N is
                    number of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
        Return:
            weight: (1+D,1) numpy array, the weights of ridge regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """

        D = xtrain.shape[1] - 1 
        reg_matrix = c_lambda * np.eye(D + 1)
        reg_matrix[0, 0] = 0.0 
        weight = np.linalg.pinv(xtrain.T@xtrain + reg_matrix)@xtrain.T@ytrain
        # print("WEIGHT SHAPE", weight.shape)
        # print(weight)
        return weight

    def ridge_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 500,
        learning_rate: float = 1e-7,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        Hints:
            - You should avoid applying regularization to the bias term in the gradient update
        """
        D = xtrain.shape[1]
        weight = np.zeros((D, 1))
        loss_per_epoch = []
        N = len(ytrain)
        for _ in range(epochs):
            y_pred = xtrain @ weight
            error = y_pred - ytrain
            # error = ytrain - y_pred

            # gradient = ((xtrain.T@error) + c_lambda * weight) / N
            reg = (c_lambda * weight) / N
            reg[0] = 0
            gradient = ((xtrain.T@error)) / N + reg
            weight -= learning_rate * gradient
            rmse = self.rmse(xtrain @ weight, ytrain)
            loss_per_epoch.append(rmse)
        return weight, loss_per_epoch


    def ridge_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.

        Hints:
            - Keep in mind that the number of epochs is the number of complete passes
            through the training dataset. SGD updates the weight for one datapoint at
            a time. For each epoch, you'll need to go through all of the points.
            - You should avoid applying regularization to the bias term in the gradient update
        """
        D = xtrain.shape[1] - 1
        weight = np.zeros((1 + D, 1))
        loss_per_step = []

        N = len(ytrain)

        for _ in range(epochs):
            for i in range(N):
                xi = xtrain[i:i+1, :]
                yi = ytrain[i:i+1]

                y_pred = xi @ weight
                error = y_pred - yi
                # print("xi.T shape: ", xi.T.shape)
                # print("error shape: ", error.shape)
                # print("weight shape: ", weight.shape)
                #  ((xtrain(NX1+D).T(1+DXN)@error(2X!+D)) + c_lambda * weight) / N
                reg = (c_lambda * weight) / N
                reg[0] = 0
                gradient =  ((xi.T@error)) + reg
                # gradient = (xi.T @ error)
                weight -= learning_rate * gradient
                rmse = self.rmse(xtrain @ weight, ytrain)
                loss_per_step.append(rmse)
        return weight, loss_per_step

    def ridge_cross_validation(
        self, X: np.ndarray, y: np.ndarray, kfold: int = 10, c_lambda: float = 100
    ) -> List[float]:  # [5 pts]
        """
        For each of the kfolds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the RMSE for each kfold

        Args:
            X : (N,1+D) numpy array, where N is the number of instances
                and D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            kfold: int, number of folds you should take while implementing cross validation.
            c_lambda: float, value of regularization constant
        Returns:
            loss_per_fold: list[float], RMSE loss for each kfold
        Hints:
            - np.concatenate might be helpful.
            - Use ridge_fit_closed for this function.
            - Look at 3.5 to see how this function is being used.
            - If kfold=10:
                split X and y into 10 equal-size folds
                use 90 percent for training and 10 percent for test
        """
        loss_per_fold = []
        fold_size = len(X) // kfold

        for i in range(kfold):
            s = i * fold_size
            e = (i + 1) * fold_size if i < kfold - 1 else len(X)

            X_test = X[s:e]
            y_test = y[s:e]
            X_train = np.concatenate((X[:s], X[e:]), axis=0)
            y_train = np.concatenate((y[:s], y[e:]), axis=0)

            weights = self.ridge_fit_closed(X_train, y_train, c_lambda)
            y_pred = self.predict(X_test, weights)

            rmse = self.rmse(y_pred, y_test)
            loss_per_fold.append(rmse)

        return loss_per_fold

    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        FUNCTION PROVIDED TO STUDENTS

        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N, 1+D) numpy array, where N is the number of instances and
                D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants (lambdas) to search from
            kfold: int, Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the average RMSE error achieved using the best_lambda
            error_list: list[float] list of average RMSE loss for each lambda value given in lambda_list
        """

        best_error = None
        best_lambda = None
        error_list = []

        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            mean_err = np.mean(err)
            error_list.append(mean_err)
            if best_error is None or mean_err < best_error:
                best_error = mean_err
                best_lambda = lm

        return best_lambda, best_error, error_list
