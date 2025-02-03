import numpy as np
from numba import jit
from numba import njit
import time
import h5py

# Kernel functions
@jit(nopython=True)
def numba_rbf(x1, x2, gamma):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    s1, _ = x1.shape
    s2, _ = x2.shape
    norm1 = np.ones((s2, 1)).dot(np.atleast_2d(np.sum(x1 ** 2, axis=1))).T
    norm2 = np.ones((s1, 1)).dot(np.atleast_2d(np.sum(x2 ** 2, axis=1)))
    return np.exp(- gamma * (norm1 + norm2 - 2 * np.dot(x1, x2.T)))

def numba_linear(x1, x2):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return np.dot(x1, x2.T)


# Objective functions
@jit(nopython=True)
def numba_dual_objective(alphas, y, K):
    a = np.sum(alphas)
    svs = np.where(alphas > 0) [0]
    
    # Compute the dot product part efficiently
    b = 0.0
    for i in svs:
        for j in svs:
            b += alphas[i] * alphas[j] * y[i] * y[j] * K[i, j]

    return a - 0.5 * b

@jit(nopython=True)
def numba_loss_val(kernel_val, alphas, y, bias, X_val, y_val):
    svs = np.where(alphas > 0) [0]
    scores = np.dot(kernel_val, (alphas[svs].flatten() * y[svs].flatten())) + bias
    losses = np.maximum(0.0, 1.0 - y_val * scores)

    total_loss = np.sum(losses)/len(X_val)
    return total_loss    

@jit(nopython=True)
def numba_primal_objective(K, alphas, y, bias, c):
    svs = np.where(alphas > 0) [0]
    n = len(alphas)
    
    # Flatten the arrays
    alphas_flat = alphas.flatten()
    y_flat = y.flatten()

    # Calculate scores
    scores = np.zeros(n)
    for i in range(n):
        score = 0.0
        for j in svs:
            score += K[i, j] * alphas_flat[j] * y_flat[j]
        scores[i] = score + bias

    # Calculate losses
    total_loss = 0.0
    for i in range(n):
        loss = max(0.0, 1.0 - y_flat[i] * scores[i])
        total_loss += loss

    # Calculate w
    w = np.zeros(n)
    for i in range(n):
        wi = 0.0
        for j in svs:
            wi += alphas_flat[j] * y_flat[j] * K[j, i]
        w[i] = wi

    # Calculate regularization term
    regularization_term = 0.0
    for i in range(n):
        regularization_term += w[i] * w[i]
    regularization_term *= 0.5
    
    return c * total_loss + regularization_term

# Optimized helper functions
@jit(nopython=True)
def numba_predict(data, alphas, kernel_val, y, bias):
    n = len(data)
    pred, scores = np.zeros(n), np.zeros(n)
    svs = np.where(alphas > 0) [0]

    scores = np.dot(kernel_val, (alphas[svs].flatten() * y[svs].flatten())) + bias
    
    pred = np.where(scores > 0, 1, -1)
    return pred, scores

@jit(nopython=True)
def accuracy_score(y_true, y_pred):
    n = len(y_true)
    correct_count = 0

    for i in range(n):
        if y_true[i] == y_pred[i]:
            correct_count += 1

    accuracy = correct_count / n
    return accuracy

@jit(nopython=True)
def numba_working_set_selection(y, alphas, G, Q, c, tau, tol, n):
    i , j = -1, -1
    G_max = -np.inf
    G_min = np.inf
    
    obj_min = np.inf
    
    for t in range(n):
        # enforce I_up criteria
        if (y[t] == 1 and alphas[t] < c) or (y[t] == -1 and alphas[t] > 0):
            # Check if the maximum objective is increased
            if -y[t] * G[t] >= G_max:
                i = t
                G_max = -y[t] * G[t]
        
    for t in range(n):
        # enforce i_down criteria
        if (y[t] == 1 and alphas[t] > 0) or (y[t] == -1 and alphas[t] < c):
            b = G_max + y[t] * G[t]
            if -y[t] * G[t] <= G_min:
                G_min = -y[t] * G[t]
            
            if b > 0:
                a1 = Q[i,i]
                a2 = Q[t, t]
                a3 = Q[i, t] * y[i] * y[t]
                
                a = a1 + a2 -2 * a3
                
                if a <= 0:
                    a = tau
                
                if (-(b * b) / a) <= obj_min:
                    j = t
                    obj_min = -(b * b) / a
    
    if G_max - G_min < tol:
        i, j = -1, -1
    
    # print('G_dif: {0}\ni: {1}\nj: {2}'.format(G_max - G_min, i, j))
    return (i, j), G_max - G_min

@jit(nopython=True)
def numba_bias(n, y, G, alphas, c):
    nr_free = 0
    ub = np.inf
    lb = -np.inf
    sum_free = 0
    
    for i in range(n):
        yG = y[i] * G[i]
        
        if alphas[i] == c:
            if y[i] == -1:
                ub = min(ub, yG)
            else:
                lb = max(lb, yG)
        elif alphas[i] == 0:
            if y[i] == +1:
                ub = min(ub, yG)
            else:
                lb = max(lb, yG)
        else:
            nr_free += 1
            sum_free += yG
            
    if nr_free > 0:
        bias = -sum_free / nr_free
    else:
        bias = -(ub + lb) / 2 
    
    return bias

# SMO class using Fan algorithm. 
class SMO_F_SVM():
    def __init__(self, 
            c = 1.0, 
            tolerance = 1e-5, 
            kernel = 'linear', 
            gamma = 1.0, 
            max_iter = -1, 
            run_iterations = 1, 
            patience = -1, 
            early_stop = False, 
            es_tolerance = 1e-5,
            objective = 'acc',
            
            log_objectives = False,
            
            debug = False
        ):
        
        self.c, self.tol, self.kernel, self.gamma  = c, tolerance, kernel, gamma
        self.max_iter, self.run_iters = max_iter, run_iterations
        
        self.w, self.bias, self.iters = 0, 0, 0
        self.examineAll, self.numchanged = 0, 0
        self.alphas, self.error_cache, self.support_vectors_ = None, None, None
        self.kernel_func = None
        
        self.G = None
        self.tau = 1e-12
        
        self.training_runs = 0
        self.no_updates = 0
        self.initialized = False
        
        self.intermediate_acc = None
        self.intermediate_loss = None
        self.intermediate_prim = None
        self.intermediate_dual = None
        self.iter_times = None
        
        self.intermediate_alphas = None
        
        self.break_flag = False
        self.obj_max:float = 0.0
        self.early_stop = early_stop
        self.patience = patience
        self.patience_cnt = 0
        self.es_tol = es_tolerance
        self.objective = objective
        self.log_objectives = log_objectives
        
        self.DEBUG = debug
        self.init_time = 0
        


    ######### Kernel Functions ##########
    def linear_kernel(self, x1, x2):
        return numba_linear(x1, x2)
    
    # k(x, y) = exp(- gamma ||x1 - x2||^2)
    # def rbf_kernel(self, gamma):
    def rbf_kernel(self, x1, x2):
        return numba_rbf(x1, x2, self.gamma)
        # return rbf_kernel


    ######### Helper Functions ##########
    def calc_Ei(self, i):
        n = len(self.alphas)
        res = np.zeros(n)
        
        for k in range(n):
            res[k] = np.dot(self.X[k, :], self.X[i, :])
        
        u = np.sum(self.alphas * self.y * res) - self.bias
        return u - self.y[i]
    
    def set_gamma(self, X):
        if isinstance(self.gamma, float):
            return self.gamma
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        elif self.gamma == 'scale':
            X_var = X.var()
            return 1.0 / (X.shape[1] * X_var) if X_var > self.eps else 1.0
        else:
            raise ValueError(f"'{self.gamma}' is incorrect value for gamma")

    def get_kernel_function(self, X):
        if callable(self.kernel):
            return self.kernel
        elif self.kernel == 'linear':
            return self.linear_kernel
        elif self.kernel == 'rbf':
            self.set_gamma(X)
            return self.rbf_kernel
        else:
            raise ValueError(f"'{self.kernel}' is incorrect value for kernel")    
        
    def update_bias(self):
        self.bias = numba_bias(self.n)
    
    def calc_objective(self):
        return numba_dual_objective(self.alphas, self.y, self.k_matrix)

    
    # Calculate the hinge loss for the validation set
    def calc_loss_val(self):
        svs = np.where(self.alphas > 0) [0]
        kernel_val = self.kernel_func(self.X_val, self.X[svs])
        return numba_loss_val(kernel_val, self.alphas, self.y, self.bias, self.X_val, self.y_val)

    
    def calc_objective_prim(self):        
        return numba_primal_objective(self.k_matrix, self.alphas, self.y, self.bias, self.c)
    
    ######### accessor functions ##########
    def reset_init(self):
        self.initialized = False  
        
    def get_training_stats(self):
        return (self.intermediate_acc, self.intermediate_loss, self.intermediate_prim, self.intermediate_dual, self.iter_times)
        
    def predict(self, data):
        svs = np.where(self.alphas > 0) [0]
        kernel_val = self.kernel_func(data, self.X[svs])

        return numba_predict(data, self.alphas, kernel_val, self.y, self.bias)
    
    def output_h5py(self, name = 'output.hdf5'):
        with h5py.File(name, 'w') as f:
            f.create_dataset('acc', data = self.intermediate_acc)
            f.create_dataset('hinge', data=self.intermediate_loss)
            f.create_dataset('primal', data=self.intermediate_prim)
            f.create_dataset('dual', data = self.intermediate_dual)
            f.create_dataset('times', data = self.iter_times)
    
    
    ######### SMO functions ##########
    def working_set_selection(self):
        result, G_dif = numba_working_set_selection(
            self.y, self.alphas, self.G, self.Q, self.c, self.tau, self.tol, self.n)
        self.G_dif = G_dif
        return result
    
    
    def init_startup(self, X, y, X_val, y_val):
        # create validation set and training set based on training data
        # Validation set is used to determine the model performance for intermediate steps. 
        init_start = time.time()
        
        self.X, self.y, self.X_val, self.y_val = X, y, X_val, y_val
        self.numchanged = 0
        self.examineAll = True
        self.n, d = self.X.shape
        
        self.alphas = np.zeros(self.n)
        self.support_vectors_ = np.zeros(self.n)   
        self.G = np.full(self.n, -1.0)
        
        self.kernel_func = self.get_kernel_function(X)
        
        self.iters = 0   
        self.training_runs = 0

        # Get the number of data points
        n = len(self.X)

        # Initialize the Q matrix with zeros
        
        self.Q = np.zeros((n, n))
        
        # Pre-compute the Q-matrix where Q[i, j] = y[i] * y[j] * K[i, j] (Fan et al. paper)
        self.k_matrix = self.kernel_func(self.X, self.X) 
        yyT = np.outer(y,y)
        
        self.Q = yyT * self.k_matrix
        

        # specify the final iteration for this training run. 
        if self.run_iters <= 0 and self.run_iters != -1:
            print("INVALID VALUE FOR RUN_ITERATIONS")
        elif self.run_iters == -1:
            self.no_updates = np.inf
        else:
            self.no_updates = self.iters + self.run_iters   
        
        self.intermediate_acc = []
        self.intermediate_acc_test = []
        self.intermediate_prim = []
        self.intermediate_loss = []
        self.intermediate_dual = []
        self.iter_times = []
        
        self.intermediate_alphas = [[] for _ in range(len(self.alphas))]
        
        if self.max_iter == -1:
            self.max_iter = np.inf

        self.obj_max = np.inf
        self.patience_cnt = self.patience
        self.break_flag = False
        
        self.init_time = time.time() - init_start
        
        self.iter_start = time.time()
        
        if self.objective == 'acc':
            self.es_delta = 1
        
        elif self.objective == 'hinge':
            self.es_delta = -1
        
        
        

    def fit(self, X, y, X_val, y_val, X_test, y_test):
        self.X_test, self.y_test = X_test, y_test
        if not self.initialized:
            self.init_startup(X, y, X_val, y_val)
            self.initialized = True
        
        while self.iters < self.max_iter:
            i, j = self.working_set_selection()
            if j == -1:
                break
            
            if self.early_stop and self.break_flag:
                print("early stopped")
                break
            
            self.iters += 1
            
            a1 = self.Q[i, i]
            a2 = self.Q[j,j]
            a3 = self.y[i] * self.y[j] * self.Q[i, j]
            
            a = a1 + a2 -2 * a3
            
            if a <= 0:
                a = self.tau
            
            b = -y[i] * self.G[i] + y[j] * self.G[j]
            
            old_alpha_i = self.alphas[i]
            old_alpha_j = self.alphas[j]
            
            self.alphas[i] += self.y[i] * b/a
            self.alphas[j] -= self.y[j] * b/a
            
            alpha_sum = self.y[i] * old_alpha_i + self.y[j] * old_alpha_j
            
            self.alphas[i] = np.clip(self.alphas[i], 0, self.c)
            self.alphas[j] = self.y[j] * (alpha_sum - self.y[i] * self.alphas[i])
            
            self.alphas[j] = np.clip(self.alphas[j], 0, self.c)
            self.alphas[i] = self.y[i] * (alpha_sum - y[j] * self.alphas[j])
            
            d_alpha_i = self.alphas[i] - old_alpha_i
            d_alpha_j = self.alphas[j] - old_alpha_j
            
            # print('i: {3}\nj: {4}\nai: {0}\naj: {1}\ngdif: {2}\n\n'.format(self.alphas[i], self.alphas[j], self.G_dif, d_alpha_i, d_alpha_j))
            
            for t in range(self.n):
                self.G[t] += self.Q[t, i] * d_alpha_i + self.Q[t, j] * d_alpha_j
     
            # start = time.time()
            if self.iters == self.no_updates:
                self.no_updates = self.iters + self.run_iters  
                self.training_runs += 1
                
                self.save_status()
                
            # print('time: {0}'.format(time.time() - start))
            # Print the alpha values for each full iteration of the algorithm. 
            if self.DEBUG:
                print("{0}\n".format(self.alphas))

            # print('time: {0}'.format(time.time() - start))

        # This code allows for studying the way alphas are handled compared to the error and kernel values. 
        # if self.DEBUG:       
        #     print("E1: {0}, E2: {1}".format(E1, E2))
        #     print("alpha1: {0}, alpha1_old: {1}".format(alpha1, alpha1_old))
        #     print("alpha2: {0}, alpha2_old: {1}".format(alpha2, alpha2_old))
        #     print("k11: {0}, k12: {1}, k22: {2}\n\n".format(k11, k12, k22))


        support = np.where((self.alphas > 0) & (self.alphas < self.c))[0]
        self.support_vectors_ = self.X[support]
        self.bias = numba_bias(self.n, self.y, self.G, self.alphas, self.c)
        

    ######### early stopping ####################  
    def save_status(self):
        # print("max: {0}, iters: {1}, no_updates: {2}, self.training_runs {3}".format(self.max_iter, self.iters, self.no_updates, self.training_runs))
        self.bias = numba_bias(self.n, self.y, self.G, self.alphas, self.c)
        # calculate all objective functions and store them in the designated arrays. 
        if self.log_objectives:
            self.intermediate_loss.append(self.calc_loss_val())
            self.intermediate_dual.append(self.calc_objective())
            self.intermediate_prim.append(self.calc_objective_prim())
            self.intermediate_acc.append(accuracy_score(self.y_val, self.predict(self.X_val)[0]))
            self.intermediate_acc_test.append(accuracy_score(self.y_test, self.predict(self.X_test)[0]))
         
            self.iter_times.append(time.time() - self.iter_start)
             
        self.iter_start = time.time()
        if self.early_stop:            
            if self.objective == 'acc':
                obj = accuracy_score(self.y_val, self.predict(self.X_val)[0])
            elif self.objective == 'hinge':
                obj = self.calc_loss_val()
            
            # print("iter: {3}\nobj: {0}\nbou: {1}\npat: {2}\ndiff: {4}\n".format(obj, self.obj_max, self.patience_cnt, self.iters, obj - self.obj_max >= 0.00001))
            if obj-self.obj_max <= self.es_delta * self.es_tol:
                self.patience_cnt -= 1
                if self.patience_cnt == 0:
                    self.break_flag = True            
            
            else:
                self.patience_cnt = self.patience
                self.obj_max = obj