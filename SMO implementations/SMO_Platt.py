import numpy as np
import random

class SMO_SVM():
    def __init__(self, c = 1.0, epsilon = 1e-5, max_iter = -1, debug = False):
        self.c = c
        self.eps = epsilon
        self.max_iter = max_iter
        
        self.w = 0
        self.bias = 0
        self.iters = 0
        
        self.alphas = []
        self.error_cache = []
        self.support_vectors_ = []
        
        self.X = 0
        self.y = 0
        
        self.DEBUG = debug

    def takestep(self, i1, i2):
        # i1 and i2 can not be the same. 
        if(i1 == i2): 
            return 0

        alpha1_old = self.alphas[i1]
        alpha2_old = self.alphas[i2]
        
        E1 = self.error_cache[i1]
        E2 = self.error_cache[i2]
        s = self.y[i1] * self.y[i2]
        
        L, H = self.calc_boundaries(i1, i2, alpha1_old, alpha2_old)
        if(L == H):
            return 0

        # we use a dot product because currently we only look at the linear kernels.
        k11 = np.dot(self.X[i1, :].T, self.X[i1, :])
        k12 = np.dot(self.X[i1, :].T, self.X[i2, :])
        k22 = np.dot(self.X[i2, :].T, self.X[i2, :])

        # Note that we use the inverse compared with the Microsoft paper. 
        eta = -2*k12 + k11 + k22
        
        # this is considered the normal situation
        if(eta > 0):
            alpha2 = alpha2_old + self.y[i2] * (E1 - E2) / eta
            
            # Clipping of alpha 2 so we work with the constrained minimum
            if alpha2 >= H:
                alpha2 = H
                
            if alpha2 <= L:
                alpha2 = L
        
        else:
            print("unusual case")
            c1 = -eta/2
            c2 = self.y[i2] * (E2 - E1) - eta*alpha2_old
            
            Lobj = c1 * L * L + c2 * L
            Hobj = c1 * H * H + c2 * H
            
            if(Lobj > Hobj):
                alpha2 = L
            elif (Lobj < Hobj):
                alpha2 = H
            else:
                alpha2 = alpha2_old
        
        # Check if enough progress is made     
        if abs(alpha2 - alpha2_old) < self.eps:
            return 0
        
        # Update alpha 1
        alpha1 = alpha1_old + s * (alpha2_old - alpha2)
 
        # Alpha 1 is not at a bound       
        if (alpha1 > 0) and (alpha1 < self.c):
            bnew = self.bias + E1 + self.y[i1] * (alpha1-alpha1_old) * (k11) + self.y[i2] *(alpha2 - alpha2_old) * k12
        # Alpha 2 is not at a bound
        elif (alpha2 > 0) and (alpha2 < self.c):
            bnew = self.bias + E2 + self.y[i2] * (alpha2-alpha2_old) * (k22) + self.y[i1] *(alpha1 - alpha1_old) * k12
        else:
            b1 = self.bias + E1 + self.y[i1] * (alpha1-alpha1_old) * (k11) + self.y[i2] *(alpha2 - alpha2_old) * k12
            b2 = self.bias + E2 + self.y[i2] * (alpha2-alpha2_old) * (k22) + self.y[i1] *(alpha1 - alpha1_old) * k12
            bnew = (b1+b2) / 2

        # This code allows for studying the way alphas are handled compared to the error and kernel values. 
        # if self.DEBUG:       
        #     print("E1: {0}, E2: {1}".format(E1, E2))
        #     print("alpha1: {0}, alpha1_old: {1}".format(alpha1, alpha1_old))
        #     print("alpha2: {0}, alpha2_old: {1}".format(alpha2, alpha2_old))
        #     print("k11: {0}, k12: {1}, k22: {2}\n\n".format(k11, k12, k22))
        
        # Use the inverse of the calculated bias. 
        self.bias = -bnew
        self.alphas[i1] = alpha1
        self.alphas[i2] = alpha2 
        
        return 1
            
    def examineExample(self, i2):
        alpha2 = self.alphas[i2]
        E2 = self.calc_Ei(i2)
        self.error_cache[i2] = E2
        
        r2 = E2 * self.y[i2]

        # Checking for the KKT conditions
        if ((r2 < -self.eps and alpha2 < self.c) or (r2 > self.eps and alpha2 > 0)):
            count = np.sum((self.alphas != 0) & (self.alphas!= self.c))
            if count > 1:
                # I1- heuristic as proposed in the paper. 
                max = 0
                i1 = -1
                for k in range(1, len(self.X)):
                    if (self.alphas[k] > 0) and (self.alphas[k] < self.c):
                        E1 = self.calc_Ei(k)
                        self.error_cache[k] = E1
                        diff = abs(E2 - E1)
                        if diff > max:
                            max = diff
                            i1 = k
                            
                if self.takestep(i1, i2):
                    return 1
                
            # Ranmdomize indexes to prevent the model from biasing towards samples at the beginning of the dataset.
            indexes = list(range(0, len(self.X)))
            random.shuffle(indexes) 
            
            
            for i in indexes:
                # First check the non-boundary examples
                if(self.alphas[i] != 0 and self.alphas[i] != self.c):
                    i1 = i
                    if self.takestep(i1, i2):
                        return 1
                    
            # Again use the set of randomized indexes and check go over all alphas.
            for j in indexes:
                i1 = j
                if self.takestep(i1, i2):
                    return 1
        return 0               
                    
    def fit(self, X, y):
        numchanged = 0
        examineAll = 1
        n, d = X.shape
        
        self.alphas = np.zeros(n)
        self.error_cache = np.zeros(n)
        self.X = X
        self.y = y
        
        while numchanged > 0 or examineAll:
            if self.iters == self.max_iter:
                break
            
            self.iters += 1
            numchanged = 0
            if examineAll:
                for i in range(n):
                    numchanged += self.examineExample(i)
            
            else:
                for i in range(n):
                    if(self.alphas[i] != 0) and (self.alphas[i] < 0):
                        numchanged += self.examineExample(i)
            
            if examineAll == 1:
                examineAll = 0
            
            elif numchanged == 0:
                examineAll = 1

            # Print the alpha values for each full iteration of the algorithm. 
            if self.DEBUG:
                print("{0}\n".format(self.alphas))
        # Round down too small alpha values and add support vectors to seperate list.
        sv_list = []
        for i in range(n):
            if self.alphas[i] < 1e-10:
                self.alphas[i] = 0
             
            if self.alphas[i] > 0:
                sv_list.append(X[i])
            
        self.support_vectors_ = np.array(sv_list)
                    
    def calc_boundaries(self, i1, i2, alpha1, alpha2):
        if self.y[i1] != self.y[i2]:
            L = max(0, alpha2 - alpha1)
            H = min(self.c, self.c + alpha2 - alpha1)
        else:
            L = max(0, alpha2+alpha1 - self.c)
            H = min(self.c, alpha2+alpha1)
        
        return L, H

    def calc_Ei(self, i):
        n = len(self.alphas)
        res = np.zeros(n)
        
        for k in range(n):
            res[k] = np.dot(self.X[k, :], self.X[i, :])
        
        u = np.sum(self.alphas * self.y * res) - self.bias
        return u - self.y[i]
    
    def predict(self, data):
        n = len(data)
        scores = np.zeros(n)
        pred = np.zeros(n)
        
        for i in range(n):
            res = np.zeros(len(self.alphas))
            
            for k in range(len(self.alphas)):
                if self.alphas[k] > 0:
                    res[k] = np.dot(self.X[k, :], data[i, :])
                    
            scores[i] = np.sum(self.alphas * self.y * res) + self.bias
            if scores[i] > 0:
                pred[i] = 1
            else:
                pred[i] = -1
        
        return pred, scores

