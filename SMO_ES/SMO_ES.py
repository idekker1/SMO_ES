"""
Module name: SMO_ES.py

This module provides the implementation of the SMO algorithm based on the implementation by Fan et al. We added functionality
for Early Stopping by implementing an extra stopping rule based on a selected objective. Additionally, we implemented the
functionality to stop training of the SVM based on a time limit.

Author: Indy Dekker
Date: 2025-03-24
"""


import numpy as np
from numba import jit
from numba import njit
import time

# Kernel functions
@jit(nopython=True)
def numba_rbf(x1, x2, gamma):
    """Calculates the Radial Basis Function (RBF) kernel value.

    Args:
        x1 (array-like): First input data point or set of points.
        x2 (array-like): Second input data point or set of points.
        gamma (float): Kernel coefficient that controls the influence of each sample.

    Returns:
        ndarray: Computed RBF kernel values between x1 and x2.
    """
    # Ensure inputs are at least 2D arrays
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    
    # Get the number of samples in each input set
    s1, _ = x1.shape
    s2, _ = x2.shape
    
    # Compute squared norms for each set of input points
    norm1 = np.ones((s2, 1)).dot(np.atleast_2d(np.sum(x1 ** 2, axis=1))).T
    norm2 = np.ones((s1, 1)).dot(np.atleast_2d(np.sum(x2 ** 2, axis=1)))
    
    # Compute the RBF kernel using the squared Euclidean distance
    return np.exp(- gamma * (norm1 + norm2 - 2 * np.dot(x1, x2.T)))

def numba_linear(x1, x2):
    """Calculates linear kernel value.

    Args:
        x1 (array-like): First input data point or set of points.
        x2 (array-like): Second input data point or set of points.

    Returns:
        ndarray: Computed linear kernel values between x1 and x2.
    """
    
    # Ensure inputs are at least 2D arrays
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    
    # Compute the linear kernel function using the dot product
    return np.dot(x1, x2.T)

# Objective functions
@jit(nopython=True)
def numba_dual_objective(alphas, y, K):
    """Calculates the dual objective function for SVM optimization.

    Args:
        alphas (array-like): Lagrange multipliers for support vectors.
        y (array-like): Labels associated with the data points.
        K (ndarray): Precomputed kernel matrix.

    Returns:
        float: Dual objective value.
    """
    # Sum of alpha values
    a = np.sum(alphas)

    # Identify support vectors (nonzero alphas)
    svs = np.where(alphas > 0)[0]

    # Compute the quadratic term in the dual objective
    b = 0.0
    for i in svs:
        for j in svs:
            b += alphas[i] * alphas[j] * y[i] * y[j] * K[i, j]

    # Return the dual objective value
    return a - 0.5 * b

@jit(nopython=True)
def numba_hinge_loss(K, alphas, y, bias, X_val, y_val):
    """Computes the hinge loss for a given validation set.

    Args:
        K (ndarray): Kernel matrix between training and validation data.
        alphas (array-like): Lagrange multipliers for support vectors.
        y (array-like): Labels associated with the training data.
        bias (float): Bias term in the SVM decision function.
        X_val (ndarray): Validation data points.
        y_val (array-like): Labels corresponding to X_val.

    Returns:
        float: Average hinge loss over the validation set.
    """
    # Identify support vectors (nonzero alphas)
    svs = np.where(alphas > 0)[0]

    # Compute the decision function scores for validation points
    scores = np.dot(K, (alphas[svs].flatten() * y[svs].flatten())) + bias

    # Compute hinge loss for each validation sample
    losses = np.maximum(0.0, 1.0 - y_val * scores)

    # Compute the mean hinge loss
    total_loss = np.sum(losses) / len(X_val)
    return total_loss
 

@jit(nopython=True)
def numba_primal_objective(K, alphas, y, bias, c):
    """Computes the primal objective function for an SVM.

    Args:
        K (ndarray): Kernel matrix.
        alphas (array-like): Lagrange multipliers for support vectors.
        y (array-like): Labels associated with the training data.
        bias (float): Bias term in the SVM decision function.
        c (float): Regularization parameter controlling trade-off between margin and loss.

    Returns:
        float: Computed primal objective value.
    """
    # Identify support vectors (nonzero alphas)
    svs = np.where(alphas > 0)[0]
    n = len(alphas)

    # Flatten the arrays for efficient computation
    alphas_flat = alphas.flatten()
    y_flat = y.flatten()

    # Compute decision function scores
    scores = np.zeros(n)
    for i in range(n):
        score = 0.0
        for j in svs:
            score += K[i, j] * alphas_flat[j] * y_flat[j]
        scores[i] = score + bias

    # Compute hinge loss term
    total_loss = 0.0
    for i in range(n):
        loss = max(0.0, 1.0 - y_flat[i] * scores[i])
        total_loss += loss

    # Compute weight vector w
    w = np.zeros(n)
    for i in range(n):
        wi = 0.0
        for j in svs:
            wi += alphas_flat[j] * y_flat[j] * K[j, i]
        w[i] = wi

    # Compute regularization term (||w||^2 / 2)
    regularization_term = 0.0
    for i in range(n):
        regularization_term += w[i] * w[i]
    regularization_term *= 0.5

    # Return the primal objective value
    return c * total_loss + regularization_term


@jit(nopython=True)
def numba_predict(data, alphas, K, y, bias):
    """Computes SVM predictions using the dual form.

    Args:
        data (ndarray): Input data points (not directly used in this function but kept for consistency).
        alphas (array-like): Lagrange multipliers for support vectors.
        K (ndarray): Kernel matrix representing similarities between support vectors and test data.
        y (array-like): Labels associated with the training data.
        bias (float): Bias term in the SVM decision function.

    Returns:
        tuple:
            - pred (ndarray): Predicted class labels (-1 or 1).
            - scores (ndarray): Raw decision scores before thresholding.
    """
    n = len(data)

    # Identify support vectors (nonzero alphas)
    svs = np.where(alphas > 0)[0]

    # Compute decision function scores
    scores = np.dot(K, (alphas[svs].flatten() * y[svs].flatten())) + bias

    # Assign class predictions based on score sign
    pred = np.where(scores > 0, 1, -1)

    return pred, scores

@jit(nopython=True)
def accuracy_score(y_true, y_pred):
    """Computes the accuracy score by comparing true and predicted labels.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Accuracy score, calculated as the proportion of correctly classified samples.
    """
    n = len(y_true)
    correct_count = 0

    # Count correct predictions
    for i in range(n):
        if y_true[i] == y_pred[i]:
            correct_count += 1

    # Compute accuracy
    accuracy = correct_count / n
    return accuracy


@jit(nopython=True)
def numba_working_set_selection(y, alphas, G, Q, c, tau, tol, n):
    """Selects a working set (pair of indices) for optimization in an SVM solver.

    Args:
        y (array-like): Labels of the dataset.
        alphas (array-like): Dual variables (Lagrange multipliers).
        G (array-like): Gradient of the objective function.
        Q (array-like): Quadratic matrix Q (Hessian).
        c (float): Regularization parameter.
        tau (float): Small positive value to prevent division by zero.
        tol (float): Tolerance for stopping criterion.
        n (int): Number of data points.

    Returns:
        tuple: Selected indices (i, j) for optimization.
        float: Difference between the maximum and minimum gradient values (G_max - G_min).
    """
    i, j = -1, -1  # Indices to be selected
    G_max = -np.inf  # Maximum gradient value
    G_min = np.inf   # Minimum gradient value
    obj_min = np.inf # Minimum objective value for selection criteria

    # Select i (working set up index) based on maximum violation
    for t in range(n):
        # I_up condition: points that can be increased
        if (y[t] == 1 and alphas[t] < c) or (y[t] == -1 and alphas[t] > 0):
            if -y[t] * G[t] >= G_max:
                i = t
                G_max = -y[t] * G[t]

    # Select j (working set down index) based on minimum objective
    for t in range(n):
        # I_down condition: points that can be decreased
        if (y[t] == 1 and alphas[t] > 0) or (y[t] == -1 and alphas[t] < c):
            b = G_max + y[t] * G[t]
            if -y[t] * G[t] <= G_min:
                G_min = -y[t] * G[t]

            if b > 0:
                a1 = Q[i, i]
                a2 = Q[t, t]
                a3 = Q[i, t] * y[i] * y[t]
                
                a = a1 + a2 - 2 * a3
                
                if a <= 0:
                    a = tau  # Avoid division by zero
                
                if (-(b * b) / a) <= obj_min:
                    j = t
                    obj_min = -(b * b) / a

    # Check stopping condition
    if G_max - G_min < tol:
        i, j = -1, -1  # No valid working set found

    return (i, j), G_max - G_min


@jit(nopython=True)
def numba_bias(n, y, G, alphas, c):
    """Computes the bias term for an SVM based on the current support vectors.

    Args:
        n (int): Number of data points.
        y (array-like): Labels of the dataset.
        G (array-like): Gradient values for each data point.
        alphas (array-like): Dual variables (Lagrange multipliers).
        c (float): Regularization parameter.

    Returns:
        float: Computed bias term.
    """
    nr_free = 0  # Number of free support vectors
    ub = np.inf  # Upper bound for bias computation
    lb = -np.inf # Lower bound for bias computation
    sum_free = 0 # Sum of free support vector gradient values

    # Iterate through all data points to compute bias
    for i in range(n):
        yG = y[i] * G[i]  # Compute signed gradient value
        
        if alphas[i] == c:  # If at upper bound
            if y[i] == -1:
                ub = min(ub, yG)
            else:
                lb = max(lb, yG)
        elif alphas[i] == 0:  # If at lower bound
            if y[i] == +1:
                ub = min(ub, yG)
            else:
                lb = max(lb, yG)
        else:  # Free support vector
            nr_free += 1
            sum_free += yG

    # Compute bias based on free support vectors or bounding values
    if nr_free > 0:
        bias = -sum_free / nr_free  # Average gradient of free support vectors
    else:
        bias = -(ub + lb) / 2  # Midpoint of upper and lower bounds
    
    return bias



class SMO_ES():
    """Class that encapsulates all the SVM functionalities with early stopping support.
    
    This class implements the Sequential Minimal Optimization (SMO) algorithm 
    with additional support for early stopping and logging intermediate results.
    
    Args:
        c (float): Regularization parameter.
        tolerance (float): Convergence tolerance.
        kernel (str): Kernel type, default is 'linear'.
        gamma (float): Kernel coefficient for 'rbf' kernel.
        max_iter (int): Maximum number of iterations (-1 for no limit).
        r (int): Number of iterations after which we check ES objectives.
        patience (int): Number of iterations without improvement before stopping (-1 for disabled).
        early_stop (bool): Whether to enable early stopping.
        es_tolerance (float): Tolerance for early stopping condition.
        es_objective (str): Optimization objective ('acc' for accuracy).
        time_lim (float, optional): Maximum allowed training time in seconds.
        log_objectives (bool): Whether to log intermediate objective values.
    """
    
    def __init__(self, 
            c=1.0, 
            tolerance=1e-3, 
            kernel='linear', 
            gamma= 'auto', 
            max_iter=-1, 
            r=1, 
            patience=-1, 
            early_stop=False, 
            es_tolerance=1e-3, 
            es_objective='acc', 
            time_lim=None, 
            log_objectives=False):
        
        # Hyperparameters and kernel settings
        self.c = c
        self.tol = tolerance
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        
        # Model parameters
        self.w = 0  # Weight vector (used if linear kernel)
        self.bias = 0  # Bias term
        self.iters = 0  # Iteration counter
        self.alphas = None  # Lagrange multipliers
        self.support_vectors_ = None  # Indices of support vectors
        self.kernel_func = None  # Kernel function
        
        # Working variables for SMO
        self.examineAll = 0  # Flag for whether to examine all points
        self.numchanged = 0  # Number of changed alphas in iteration
        self.error_cache = None  # Cache for gradient values
        self.G = None  # Gradient array
        self.tau = 1e-12  # Small value for numerical stability
        
        # Early stopping variables
        self.es_objective = es_objective # set
        self.early_stop = early_stop  # Flag to enable early stopping
        self.patience = patience  # Patience counter for early stopping
        self.patience_cnt = 0  # Counter for patience
        self.es_tol = es_tolerance  # Tolerance for early stopping
        self.r = r
        self.obj_max = 0.0  # Stores max objective value for stopping

        
        # Logging and tracking
        self.iters = 0 # Number of iterations of the SMO algorithm
        self.ES_iters = 0  # Number of executed objective assesments
        self.no_updates = 0  # Counter for unchanged solutions
        self.initialized = False  # Whether the model has been initialized
        
        # Status variables
        self.finished = False
        self.ES_flag = False
               
        # Logging settings
        self.log_objectives = log_objectives  # Whether to log objectives
        
        # Time limit
        self.time_lim = time_lim  # Maximum training time
        self.init_time = 0  # Time at initialization
             
    def linear_kernel(self, x1, x2):
        """Calculates linear kernel value using an numba optimized function.

        Args:
            x1 (array-like): First input data point or set of points.
            x2 (array-like): Second input data point or set of points.

        Returns:
            ndarray: Computed linear kernel values between x1 and x2.
        """
        return numba_linear(x1, x2)
    
    def rbf_kernel(self, x1, x2):
        """Calculates the Radial Basis Function (RBF) kernel value using an numba optimized function.

        Args:
            x1 (array-like): First input data point or set of points.
            x2 (array-like): Second input data point or set of points.
            gamma (float): Kernel coefficient that controls the influence of each sample.

        Returns:
            ndarray: Computed RBF kernel values between x1 and x2.
        """
        return numba_rbf(x1, x2, self.gamma)

    def set_gamma(self, X):
        """Determines the gamma value for the model based on the specified configuration.

        Args:
            X (array-like): Input data used to calculate the gamma value. Should be a 2D array where rows represent samples and columns represent features.

        Returns:
            float: The computed gamma value based on the configuration of `self.gamma`.

        Raises:
            ValueError: If `self.gamma` is set to a value other than 'auto', 'scale', or a float.
        """
        if isinstance(self.gamma, float):
            return self.gamma
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        elif self.gamma == 'scale':
            X_var = X.var()
            return 1.0 / (X.shape[1] * X_var) if X_var > self.tol else 1.0
        else:
            raise ValueError(f"'{self.gamma}' is incorrect value for gamma")

    def get_kernel_function(self, X):
        """Returns the appropriate kernel function based on the specified kernel configuration.

        Args:
            X (array-like): Input data used for kernel calculation. Typically a 2D array where rows represent samples and columns represent features.

        Returns:
            callable: A function representing the kernel to be used in the model. The function is one of `self.kernel`, `linear_kernel`, or `rbf_kernel`.

        Raises:
            ValueError: If `self.kernel` is set to a value other than 'linear', 'rbf', or a callable kernel function.
        """
        if callable(self.kernel):
            return self.kernel
        elif self.kernel == 'linear':
            return self.linear_kernel
        elif self.kernel == 'rbf':
            self.gamma = self.set_gamma(X)
            return self.rbf_kernel
        else:
            raise ValueError(f"'{self.kernel}' is incorrect value for kernel")

    def calc_dual_objective(self):
        """Calculates the dual objective function for SVM optimization using a numba optimed function.
        
        This functin computes the dual objective using a numba-optimized function based on teh kernel matrix,
        the Lagrange multipliers (alphas) and the target labels.

        Returns:
            float: Computed dual objective value.
        """
        return numba_dual_objective(self.alphas, self.y, self.k_matrix)

    def calc_hinge_loss(self):
        """Calculates the hinge loss for the validation set.

        This function identifies the support vectors, computes the kernel values between the validation set and support vectors, 
        and then calculates the hinge loss using a numba-optimized function.

        Returns:
            float: The computed hinge loss value for the validation set.
        """
        svs = np.where(self.alphas > 0)[0]
        kernel_val = self.kernel_func(self.X_val, self.X[svs])
        return numba_hinge_loss(kernel_val, self.alphas, self.y, self.bias, self.X_val, self.y_val)

    def calc_primal_objective(self):
        """Calculates the primal objective for the optimization problem.

        This function computes the primal objective value using a numba-optimized function based on the kernel matrix,
        the Lagrange multipliers (alphas), the target labels, the bias term, and the regularization parameter.

        Returns:
            float: The computed primal objective value.
        """
        return numba_primal_objective(self.k_matrix, self.alphas, self.y, self.bias, self.c)
            
    def predict(self, data):
        """Makes predictions on the given data using the trained model.

        This function computes the kernel values between the input data and the support vectors, 
        and then uses a numba-optimized function to predict the labels based on the support vectors and model parameters.

        Args:
            data (array-like): Input data for which predictions are to be made. Typically a 2D array where rows represent samples and columns represent features.

        Returns:
            ndarray: The predicted labels for the input data.
        """
        svs = np.where(self.alphas > 0)[0]
        kernel_val = self.kernel_func(data, self.X[svs])
        return numba_predict(data, self.alphas, kernel_val, self.y, self.bias)
   
    def working_set_selection(self):
        """Selects the working set for the optimization problem.

        This function utilizes a numba-optimized function to perform working set selection based on the 
        current values of the Lagrange multipliers (alphas), the gradient (G), the kernel matrix (Q), 
        and other model parameters. It updates the working set and stores the difference in the gradients.

        Returns:
            ndarray: The selected working set.
        """
        result, G_dif = numba_working_set_selection(
            self.y, self.alphas, self.G, self.Q, self.c, self.tau, self.tol, self.n)
        self.G_dif = G_dif
        return result

    
    def init_startup(self, X, y, X_val, y_val):
        """Initializes the SVM model with training and validation data.

        This method prepares the kernel matrix, gradient values, and key parameters
        for the SMO optimization process.
        
        Args:
            X (ndarray): Training data of shape (n_samples, n_features).
            y (ndarray): Training labels of shape (n_samples,).
            X_val (ndarray): Validation data for intermediate evaluations.
            y_val (ndarray): Validation labels.
        """
        # Store dataset
        self.X, self.y = X, y
        self.X_val, self.y_val = X_val, y_val
        
        # Initialize parameters
        self.n, _ = self.X.shape  # Number of samples, features
        self.alphas = np.zeros(self.n)
        self.support_vectors_ = np.zeros(self.n)   
        self.G = np.full(self.n, -1.0)  # Gradient vector initialization

        self.numchanged = 0
        self.examineAll = True  # Start by examining all points
        self.iters = 0
        self.ES_iters = 0  # Tracks iterations for early stopping

        # Assign kernel function
        self.kernel_func = self.get_kernel_function(X)
        
        # Precompute the Q-matrix (Q[i, j] = y[i] * y[j] * K[i, j] from Fan et al.)
        self.k_matrix = self.kernel_func(self.X, self.X)  
        yyT = np.outer(y, y)
        self.Q = yyT * self.k_matrix 

        # Handle maximum iterations
        if self.max_iter == -1:
            self.max_iter = np.inf

        # Initialize early stopping settings
        self.patience_cnt = self.patience  # Start patience counter
        
        # Define improvement direction for early stopping
        if self.es_objective == 'acc':  # Maximize accuracy
            self.es_delta = 1
            self.obj_max = -np.inf
        elif self.es_objective == 'hinge':  # Minimize hinge loss
            self.es_delta = -1
            self.obj_max = np.inf


        # Set time limit if specified
        if self.time_lim is not None:
            self.end_time = time.time() + self.time_lim

        # Initialize logs if required
        if self.log_objectives:
            self.intermediate_acc = []
            self.intermediate_prim = []
            self.intermediate_hinge = []
            self.intermediate_dual = []
     
        
        
    def fit(self, X, y, X_val = None, y_val = None):
        """Trains the SVM using the SMO algorithm with early stopping."""
        
        # Initialize if not already done
        if not self.initialized:
            # Need validation data to log objectives or apply early stopping.
            if (self.log_objectives or self.early_stop) and (X_val is None or y_val is None):
                raise ValueError(f"Terminate training: X_val or y_val is None")
            
            self.init_startup(X, y, X_val, y_val)
            self.initialized = True
        
        while self.iters < self.max_iter:
            # Select working set
            i, j = self.working_set_selection()
            if j == -1:
                self.finished = True # Training SVM is finished
                break  # Convergence reached
            
            # Compute Q-related terms
            a1 = self.Q[i, i]
            a2 = self.Q[j, j]
            a3 = self.y[i] * self.y[j] * self.Q[i, j]
            a = a1 + a2 - 2 * a3
            
            if a <= 0:
                a = self.tau  # Avoid division by zero
            
            b = -y[i] * self.G[i] + y[j] * self.G[j]
            
            # Store old alpha values
            old_alpha_i, old_alpha_j = self.alphas[i], self.alphas[j]
            
            # Update alpha values
            self.alphas[i] += self.y[i] * b / a
            self.alphas[j] -= self.y[j] * b / a
            
            # Maintain sum constraint
            alpha_sum = self.y[i] * old_alpha_i + self.y[j] * old_alpha_j
            
            # Clip alphas to valid range
            self.alphas[i] = np.clip(self.alphas[i], 0, self.c)
            self.alphas[j] = self.y[j] * (alpha_sum - self.y[i] * self.alphas[i])
            self.alphas[j] = np.clip(self.alphas[j], 0, self.c)
            self.alphas[i] = self.y[i] * (alpha_sum - y[j] * self.alphas[j])
            
            # Compute alpha changes
            d_alpha_i = self.alphas[i] - old_alpha_i
            d_alpha_j = self.alphas[j] - old_alpha_j
            
            # Update gradient G
            for t in range(self.n):
                self.G[t] += self.Q[t, i] * d_alpha_i + self.Q[t, j] * d_alpha_j
            
            
            # Check if ES objectives need to be checked.
            if self.iters == self.no_updates:
                self.no_updates = self.iters + self.r  
                self.ES_iters += 1
                self.bias = numba_bias(self.n, self.y, self.G, self.alphas, self.c)
                
                # Log objectives
                if self.log_objectives:
                    self.intermediate_hinge.append(self.calc_hinge_loss())
                    self.intermediate_dual.append(self.calc_dual_objective())
                    self.intermediate_prim.append(self.calc_primal_objective())
                    self.intermediate_acc.append(accuracy_score(self.y_val, self.predict(self.X_val)[0]))

                if self.early_stop:
                    # Check if early stopping is based on accuracy
                    if self.es_objective == 'acc':
                        # Compute the current accuracy on the validation set
                        obj = accuracy_score(self.y_val, self.predict(self.X_val)[0])
                        
                        # If the improvement is smaller than the tolerance threshold
                        if obj - self.obj_max <= self.es_delta * self.es_tol:
                            self.patience_cnt -= 1  # Decrease patience count
                            
                            # Stop training if patience runs out
                            if self.patience_cnt == 0:
                                print("Training Stopped Early")
                                self.ES_flag = True  # Set early stopping flag
                                self.finished = True # Training SVM is finished
                                break
                        else:
                            # Reset patience count and update the best observed accuracy
                            self.patience_cnt = self.patience
                            self.obj_max = obj                    
                    
                    # Check if early stopping is based on hinge loss
                    if self.es_objective == 'hinge':
                        # Compute the current hinge loss
                        obj = self.calc_hinge_loss()
                        
                        # If the deterioration exceeds the tolerance threshold
                        if obj - self.obj_max >= self.es_delta * self.es_tol:
                            self.patience_cnt -= 1  # Decrease patience count
                            
                            # Stop training if patience runs out
                            if self.patience_cnt == 0:
                                print("Training Stopped Early")
                                self.ES_flag = True  # Set early stopping flag
                                self.finished = True # Training SVM is finished
                                break
                        else:
                            # Reset patience count and update the best observed hinge loss
                            self.patience_cnt = self.patience
                            self.obj_max = obj
  

            # Check time limit
            if self.time_lim is not None and time.time() > self.end_time:
                print('Training Stopped: Max training time exceeded')
                break
            
            self.iters += 1  # Move to next iteration

        # After finishing the main loop we mark the support vectors
        support = np.where((self.alphas > 0) & (self.alphas < self.c))[0]
        self.support_vectors_ = self.X[support]
        
        # calculate the bias of the model.
        self.bias = numba_bias(self.n, self.y, self.G, self.alphas, self.c)
        self.finished = True # Training SVM is finished
        

