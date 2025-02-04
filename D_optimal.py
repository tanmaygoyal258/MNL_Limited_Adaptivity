import numpy as np
from utils import information_matrix_set , weighted_norm

class D_Optimal():
    def __init__(self , dim , arms , gamma = 1e-4 , delta = 5e-4):

        self.arms = arms
        self.dim = dim

        self.spanning_set= self.arms.copy()
        self.weights = [np.random.random()for _ in range(len(arms))]
        self.weights /= np.sum(self.weights)
        self.weights = self.weights.tolist()

        self.gamma = gamma
        self.delta = delta

        self.construct_spanner()
        self.d_optimal_policy = self.expand_spanner_to_normal_length()

    def construct_spanner(self):
        '''
        construct the spanning set and the weights
        '''
        while len(self.spanning_set) > self.dim:
            current_weights = self.weights
            while True:
                D = np.linalg.inv(information_matrix_set(current_weights , self.spanning_set ))
                new_weights = [w/self.dim * weighted_norm(arm , D)**2 for w,arm in zip(current_weights , self.spanning_set)]
                error = np.sum(np.abs(np.sum(new_weights.copy()) - np.sum(current_weights.copy())))
                if error < self.gamma:
                    break
                else:
                    current_weights = new_weights
            
            final_weights = []
            final_arms = []
            for w , arm in zip(new_weights , self.spanning_set):
                if w > self.delta:
                    final_weights.append(w)
                    final_arms.append(arm)
            self.spanning_set = final_arms
            self.weights = final_weights
            if final_arms == self.spanning_set:
                break


    def expand_spanner_to_normal_length(self):
        '''
        return an array of size len(arms) for the distribution
        '''
        distribution = [0 for _ in range(len(self.arms))]
        self.spanning_set_tuple = [tuple(a) for a in self.spanning_set]
        for idx , arm in enumerate(self.arms):
            if tuple(arm) in self.spanning_set_tuple:
                i = self.spanning_set_tuple.index(tuple(arm))
                distribution[idx] = self.weights[i]

        distribution = (distribution/np.sum(distribution)).tolist()

        return distribution