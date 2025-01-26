import numpy as np
from utils import information_matrix_set , weighted_norm
from utils import g_optimal_design

class CoreIdentification():
    def __init__(self , lamda , c , multiset , dim):
        self.lamda = lamda
        self.c = c
        self.multiset = multiset
        self.dim = dim

        self.current_set = multiset

        self.gamma = len(self.multiset)

    def findCoreSet(self):
        '''
        finds the core set using the algorithm in the paper
        '''
        while True:
            avg_design_matrix = self.findAvgDesignMatrix()
            avg_design_matrix_inv = np.linalg.inv(avg_design_matrix)
            list_of_max = []
            for X in self.current_set:
                max_in_X = -np.inf
                for x in X:
                    max_in_X = max(max_in_X , weighted_norm(x , avg_design_matrix_inv)**2)
                list_of_max.append(max_in_X)
            if np.max(list_of_max) <= self.dim**self.c:
                return self.current_set
            else:
                self.current_set = [self.current_set[i] for i in range(len(self.current_set)) if list_of_max[i] <= (self.dim**self.c)/2]

    def findAvgDesignMatrix(self):
        '''
        finds the average design matrix for the core set
        '''
        avg_design_matrix = np.identity(self.dim) * self.lamda
        for X in self.current_set:
            g_opt_design = g_optimal_design(X)
            avg_design_matrix += 1/self.gamma * information_matrix_set(X , g_opt_design)
        return avg_design_matrix