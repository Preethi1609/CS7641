import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = True # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """
        max_logit = np.max(logit,axis=1,keepdims=True)
        new_logit = logit - max_logit
        prob = np.exp(new_logit)/(np.sum(np.exp(new_logit),axis=1,keepdims=True)) 
        return prob

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        max_logit = np.max(logit, axis=1, keepdims=True)
        s = np.log(np.sum(np.exp(logit-max_logit),axis=1, keepdims=True))
        s += max_logit
        return s

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """

        raise NotImplementedError

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """
        N, D = points.shape
        inverse_sigma = np.linalg.inv(sigma_i)
        determinant_sigma = np.linalg.det(sigma_i)
        num = 1 / ((2 * np.pi) ** (D / 2) * np.sqrt(determinant_sigma))
        power = -0.5 * np.sum((points - mu_i) @ inverse_sigma * (points - mu_i), axis=1)

        return num * np.exp(power)

    def create_pi(self):
        """
        Initialize the prior probabilities 
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        return np.full(shape=self.K,fill_value = 1/self.K)

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """

        # random_indices = [int(np.random.uniform(0, self.points.shape[0], replace=False)) for _ in range(self.K)]
        # random_indices = (np.random.uniform(0, self.points.shape[0], self.K))
        random_indices = (np.random.random_integers(0, self.points.shape[0]-1, self.K))
        # print("random indices: ", random_indices)
        mu_initial = self.points[random_indices, :]
        return mu_initial
    
    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the 
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        diagonal_matrix = np.eye(self.D)
        # print(diagonal_matrix)
        repeat = np.tile(diagonal_matrix, (self.K, 1, 1))
        # print(repeat)
        return repeat
    
    def _init_components(self, **kwargs):  # [5pts]

        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) may be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5) #Do Not Remove Seed    
        pi = self.create_pi()
        mu = self.create_mu()       
        sigma =  self.create_sigma()
        # print(pi, mu, sigma)
        return pi,mu,sigma


        raise NotImplementedError

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        dim = self.points.shape
        N = dim[0]
        D = dim[1]
        K = mu.shape[0]
        
        ll = []
        
        for i in range(K):
            prev_pi = pi[i]
            normalize_points = (self.points-mu[i])
            determinant_sigma = np.linalg.det(sigma[i])
            inverse_sigma = np.linalg.inv(sigma[i])
            constant_term = 1/np.sqrt(((2*np.pi)**D)*determinant_sigma)
            quadratic_forms = np.einsum('ij,jk,ki->i', normalize_points, inverse_sigma, normalize_points.T)
            pdf_values = constant_term * np.exp(-0.5 * quadratic_forms)
            pdf = np.log(pdf_values + 1e-32)
            lli = np.log(prev_pi + 1e-32) + pdf        
            ll.append(lli)
        ll = np.array(ll).T
        return ll


    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        log_likelihood = self._ll_joint(pi, mu, sigma)
        soft_cluster_assignment= self.softmax(log_likelihood)
        return soft_cluster_assignment

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        dim = self.points.shape
        N = dim[0]
        D = dim[1]
        K = gamma.shape[1]
        cluster_means = np.zeros((K, D))
        cluster_covariances = np.zeros((K, D, D))
        cluster_priors = np.zeros(K)
        for i in range(K):
            gamma_i = gamma[:, i]

            mu_i = np.sum(self.points * gamma_i.reshape(N, 1), axis=0) / np.sum(gamma_i)
            normalized_pts = self.points - mu_i
            cluster_means[i] = (mu_i)

            sigma_i = np.dot((gamma_i.reshape(N, 1) * (normalized_pts)).T, (normalized_pts)) / np.sum(gamma_i)
            cluster_covariances[i] = (sigma_i)

            pi_i = np.sum(gamma_i) / N
            cluster_priors[i] = (pi_i)

        return cluster_priors, cluster_means, cluster_covariances
        
    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)

