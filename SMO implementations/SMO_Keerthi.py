# Inspired on this implementation https://github.com/maik-nack/SMO/blob/master/SVM.py


import numpy as np

from sklearn.metrics import accuracy_score

class SMO_K_SVM():
    def __init__(self, 
            c = 1.0, 
            tolerance = 1e-5, 
            epsilon = 0, 
            kernel = 'linear', 
            gamma = 1.0, 
            max_iter = -1, 
            run_iterations = -1, 
            patience = -1, 
            early_stop = False, 
            es_tolerance = 1e-5,
            debug = False
        ):
        
        self.c, self.tol, self.eps, self.kernel, self.gamma  = c, tolerance, epsilon, kernel, gamma
        self.max_iter, self.run_iters = max_iter, run_iterations
        
        self.w, self.bias, self.iters = 0, 0, 0
        self.examineAll, self.numchanged = 0, 0
        self.alphas, self.error_cache, self.support_vectors_ = None, None, None
        self.kernel_func = None
        
        
        self.I_0, self.I_1, self.I_2, self.I_3, self.I_4 = None, None, None, None, None
        
        self.training_runs = 0
        self.no_updates = 0
        self.initialized = False
        
        self.intermediate_acc = None
        
        self.break_flag = False
        self.obj_max:float = 0.0
        self.early_stop = early_stop
        self.patience = patience
        self.patience_cnt = 0
        self.es_tol = es_tolerance
        
        self.intermediate_acc = None
        
        self.DEBUG = debug


    ######### Kernel Functions ##########
    def linear_kernel(self, x1, x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        return x1.dot(x2.T)
    
    # k(x, y) = exp(- gamma ||x1 - x2||^2)
    # def rbf_kernel(self, gamma):
    def rbf_kernel(self, x1, x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        s1, _ = x1.shape
        s2, _ = x2.shape
        norm1 = np.ones((s2, 1)).dot(np.atleast_2d(np.sum(x1 ** 2, axis=1))).T
        norm2 = np.ones((s1, 1)).dot(np.atleast_2d(np.sum(x2 ** 2, axis=1)))
        return np.exp(- self.gamma * (norm1 + norm2 - 2 * x1.dot(x2.T)))
        # return rbf_kernel


    ######### Helper Functions ##########
    def calc_boundaries(self, i1, i2, alpha1, alpha2):
        if self.y[i1] != self.y[i2]:
            L = max(0, alpha2 - alpha1)
            H = min(self.c, self.c + alpha2 - alpha1)
        else:
            L = max(0, alpha2+alpha1 - self.c)
            H = min(self.c, alpha2+alpha1)
        
        return L, H


    def calc_Fi(self, i):
        return np.sum(self.kernel_func(self.X[i], self.X) * (self.alphas * self.y)) - self.y[i]
    
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
    
    # def calc_objective(self):
    #     return np.sum(self.alphas) - 0.5 * np.dot(self.alphas, np.dot(self.alphas * self.y, self.kernel_func(self.X, self.X)))

    # def calc_objective(self):
    #     a = np.sum(self.alphas)
    #     b = 0
    #     for i in range(len(self.alphas)):
    #         for j in range(len(self.alphas)):
    #             b += self.alphas[i] * self.alphas[j] *  self.y[i] * self.y[j] * self.kernel_func(self.X[i],self.X[j])
    #     print(a, b)       
    #     return a - 0.5 * b[0][0]
    
    def calc_objective(self):
        a = np.sum(self.alphas)
        b = np.sum(self.alphas[:, None] * self.alphas * self.y[:, None] * self.y * self.kernel_func(self.X, self.X))
        return a - 0.5 * b
    
    ######### Callable functions ##########
    def reset_init(self):
        self.initialized = False  
        
    def predict(self, data):
        n = len(data)
        pred, scores = np.zeros(n), np.zeros(n)
        
        for i in range(n):
            scores[i] = np.sum(self.kernel_func(data[i], self.X) * (self.alphas * self.y)) + self.bias
            if scores[i] > 0:
                pred[i] = 1
            else:
                pred[i] = -1
        return pred, scores
    
    
    ######### SMO Functions ##########
    def update_I(self, i):
        alpha = self.alphas[i]
        if self.I_0[i]:
            self.I_0[i] = False
        else:
            if self.y[i] == 1:
                if self.I_1[i]:
                    self.I_1[i] = False
                else:
                    self.I_3[i] = False
            else:
                if self.I_2[i]:
                    self.I_2[i] = False
                else:
                    self.I_4[i] = False
                    
        if alpha <= self.eps or alpha >= self.c - self.eps:
            if self.y[i] == 1:
                if alpha <= self.eps:
                    self.I_1[i] = True
                else:
                    self.I_3[i] = True
            else:
                if alpha <= self.eps:
                    self.I_4[i] = True
                else:
                    self.I_2[i] = True
        else:
            self.I_0[i] = True
            
            
    def update_I_low_up(self, I_low, I_up, i):
        if self.I_3[i] or self.I_4[i]:
            I_low[i] = True
        else:
            I_up[i] = True
            

    def get_b_i(self, I, argfunc):
        I = np.where(I)[0]
        F = self.Fcache[I]
        i = I[argfunc(F)]
        b = self.Fcache[i]
        return b, i
    
    
    def takestep(self, i1, i2):
        # i1 and i2 can not be the same. 
        if(i1 == i2): 
            return 0

        alpha1_old = self.alphas[i1]
        alpha2_old = self.alphas[i2]
        s = self.y[i1] * self.y[i2]
        F1 = self.Fcache[i1]
        F2 = self.Fcache[i2]
        
                
        L, H = self.calc_boundaries(i1, i2, alpha1_old, alpha2_old)
        if abs(L-H) < self.eps:
            # print("l == h")
            return 0

        
        # we use a dot product because currently we only look at the linear kernels.
        k11 = self.kernel_func(self.X[i1], self.X[i1])[0,0]
        k12 = self.kernel_func(self.X[i1], self.X[i2])[0,0]
        k22 = self.kernel_func(self.X[i2], self.X[i2])[0,0]

        # Note that we use the inverse compared with the Microsoft paper. 
        eta = 2*k12 - k11 - k22
        
        # this is considered the normal situation
        if(eta < -self.eps):
            alpha2 = alpha2_old - self.y[i2] * (F1 - F2) / eta
            
            # Clipping of alpha 2 so we work with the constrained minimum
            if alpha2 > H:
                alpha2 = H
                
            elif alpha2 < L:
                alpha2 = L
        
        else:
            c1 = eta/2
            c2 = self.y[i2] * (F1 - F2) - eta*alpha2_old
            
            Lobj = c1 * L * L + c2 * L
            Hobj = c1 * H * H + c2 * H
            
            if(Lobj > Hobj):
                alpha2 = L
            elif (Lobj < Hobj):
                alpha2 = H
            else:
                alpha2 = alpha2_old
        
        # Check if enough progress is made     
        if abs(alpha2 - alpha2_old) < self.eps * (alpha2 + alpha2_old + self.eps):
            return 0
        
        # Update alpha 1
        alpha1 = alpha1_old + s * (alpha2_old - alpha2)
        
        d_alpha1 = alpha1 - alpha1_old
        d_alpha2 = alpha2 - alpha2_old

        ki1 = self.kernel_func(self.X[self.I_0], self.X[i1]).ravel()
        ki2 = self.kernel_func(self.X[self.I_0], self.X[i2]).ravel()
        self.Fcache[self.I_0] += self.y[i1] * (d_alpha1) * ki1 + self.y[i2] * (d_alpha2) * ki2

        # Round numerical errors that arise due to the calculation of Fi        
        if alpha1 < self.c * 1e-3:
            alpha1 = 0
        elif alpha1 > self.c - self.c * 1e-3:
            alpha1 = self.c
            
        if alpha2 < self.c * 1e-3:
            alpha2 = 0
        elif alpha2 > self.c - self.c * 1e-3:
            alpha2 = self.c
        
        # update alpha vector
        self.alphas[i1] = alpha1
        self.alphas[i2] = alpha2

                      
        self.update_I(i1)
        self.update_I(i2)

        self.Fcache[i1] = F1 + self.y[i1] * d_alpha1 * k11 + self.y[i2] * d_alpha2 * k12
        self.Fcache[i2] = F2 + self.y[i1] * d_alpha1 * k12 + self.y[i2] * d_alpha2 * k22

        # Compute (i_low, b_low) and (i_up, b_up)
        I_low, I_up = self.I_0.copy(), self.I_0.copy()
        self.update_I_low_up(I_low, I_up, i1)
        self.update_I_low_up(I_low, I_up, i2)
        self.b_down, self.i_down = self.get_b_i(I_low, np.argmax)
        self.b_up, self.i_up = self.get_b_i(I_up, np.argmin)
                
        # update the number of iterations after an update of the alphas has taken place. 
        self.iters += 1
        
        return 1
    
           
    def examineExample(self, i1):
        if self.I_0[i1]:
            F1 = self.Fcache[i1]

        else:
            F1 = self.calc_Fi(i1)
            self.Fcache[i1] = F1

            if (self.I_1[i1] or self.I_2[i1]) and F1 < self.b_up:
                self.b_up = F1
                self.i_up = i1
            
            elif(self.I_3[i1] or self.I_4[i1]) and F1 > self.b_down:
                self.b_down = F1
                self.i_down = i1

        # check optimality using the current values of b_low and b_up.
        # if a violation occurs we update i2 for the update step. 
        # See equations 2.9a and 2.9b from Keerthi Paper.
        optimality = True
        i2 = 0

        if (self.I_0[i1] or self.I_1[i1] or self.I_2[i1]) and self.b_down - F1 > 2*self.tol:
            optimality = False
            i2 = self.i_down

        # check if i1 is in I_down
        if (self.I_0[i1] or self.I_3[i1] or self.I_4[i1]) and F1 - self.b_up > 2 * self.tol:
            optimality = False
            i2 = self.i_up
        
        if optimality:
            return 0

        
        if self.I_0[i1]:
            if self.b_down - F1 > F1 - self.b_up:
                i2 = self.i_down
            else:
                i2 = self.i_up
        
        return self.takestep(i2, i1)
      
      
    def init_startup(self, X, y, X_val, y_val):
        # create validation set and training set based on training data
        # Validation set is used to determine the model performance for intermediate steps. 
        
        self.X, self.y, self.X_val, self.y_val = X, y, X_val, y_val
        self.numchanged = 0
        self.examineAll = True
        self.n, d = self.X.shape
        
        self.alphas = np.zeros(self.n)
        self.error_cache = np.zeros(self.n)
        
        self.b_up, self.b_down = -1,1
        
        y1 = y == 1
        self.i_up = y1.nonzero()[0][0]
        self.I_1 = y1
        
        y2 = y == -1
        self.i_down = y2.nonzero()[0][0]
        self.I_4 = y2
        
        self.I_0, self.I_2, self.I_3 = np.zeros(self.y.shape, bool), np.zeros(self.y.shape, bool), np.zeros(self.y.shape, bool)
        
        self.Fcache = np.zeros(self.n)
        self.Fcache[self.i_up] = -1
        self.Fcache[self.i_down] = 1
        
        self.support_vectors_ = np.zeros(self.n)   
        
        self.kernel_func = self.get_kernel_function(X)
        
        self.iters = 0   
        self.training_runs = 0
        
        # specify the final iteration for this training run. 
        if self.run_iters <= 0 and self.run_iters != -1:
            print("INVALID VALUE FOR RUN_ITERATIONS")
        elif self.run_iters == -1:
            self.no_updates = np.inf
        else:
            self.no_updates = self.iters + self.run_iters   
        
        self.patience_cnt = self.patience
        self.break_flag = False
        self.intermediate_acc = []

            
                
    def fit(self, X, y, X_val, y_val):
        if not self.initialized:
            self.init_startup(X, y, X_val, y_val)
            self.initialized = True
              
        while (self.numchanged > 0 or self.examineAll):
            if(self.iters >= self.max_iter and self.max_iter != -1) or self.break_flag:
                break
            
            self.numchanged = 0
            if self.examineAll:
                for i in range(self.n):
                    self.numchanged += self.examineExample(i)
                                     
                    if self.iters == self.no_updates:
                        self.save_status()  
                    # after returning we check if we did met the max number of iterations for a single training_run
                    if  (self.iters >= self.max_iter and self.max_iter != -1) or self.break_flag:
                        break
            
            else:
                while True:
                    i2 = self.i_down
                    i1 = self.i_up
                    inner_loop_success = self.takestep(i1,i2)
                    
                    if self.iters == self.no_updates:
                        self.save_status()  

                    self.numchanged += inner_loop_success
                
                    if self.b_up > self.b_down - 2 * self.tol or not inner_loop_success:
                        break
                
                    # after returning we check if we did met the max number of iterations for a single training_run
                    if (self.iters >= self.max_iter and self.max_iter != -1) or self.break_flag:
                        break
                self.numchanged = 0
                
            if self.examineAll:
                self.examineAll = False
            
            elif self.numchanged == 0:
                self.examineAll = True


            # Print the alpha values for each full iteration of the algorithm. 
            if self.DEBUG:
                print("{0}\n".format(self.alphas))
        
        # Round down extremely small alpha values for stability
        tmp = np.where(self.alphas < self.c * 1e-3)[0]
        self.alphas[tmp] = 0
        
        support = np.where(self.alphas > self.eps)[0]
        self.support_vectors_ = self.X[support]
        
        self.bias = -(self.b_down + self.b_up) / 2
        
    
      
    ######### early stopping ####################  
    def save_status(self):
        # print("max: {0}, iters: {1}, no_updates: {2}, self.training_runs {3}".format(self.max_iter, self.iters, self.no_updates, self.training_runs))
        # self.intermediate_acc.append(accuracy_score(self.y_val, self.predict(self.X_val)[0]))
        obj = self.calc_objective()
        self.intermediate_acc.append(obj)
        
        if self.early_stop:
            
            # print("iter: {3}\nobj: {0}\nbou: {1}\npat: {2}\ndiff: {4}\n".format(obj, self.obj_max, self.patience_cnt, self.iters, obj - self.obj_max >= 0.00001))
            
            if obj <= self.obj_max  + self.es_tol:
                self.patience_cnt -= 1
                if self.patience_cnt == 0:
                    self.break_flag = True            
            
            else:
                self.patience_cnt = self.patience
                self.obj_max = obj     
        # self.no_updates to specify the next training run. 
        self.no_updates = self.iters + self.run_iters  
        self.training_runs += 1
                

