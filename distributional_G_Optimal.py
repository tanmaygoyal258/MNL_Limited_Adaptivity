import numpy as np
from tqdm import tqdm
from CoreIdentification import CoreIdentification
from utils import obtainMixedSoftmaxParameters , g_optimal_design , information_matrix_set , sample_softmax

class Distributional_G_optimal():

    def __init__(self):

        self.p_array = None
        self.matrices = None

    def update_parameters(self , lamda , S , dim_arms):
        '''
        finds the probability and matrices for the mixed softmax policy
        '''

        c = 6
        core_set = CoreIdentification(lamda , c , S , dim_arms).findCoreSet()
        self.p_array , self.matrices =  obtainMixedSoftmaxParameters(core_set , lamda , dim_arms)

    def obtainMixedSoftmaxParameters(self , S , lamda , d):
        '''
        returns the mixed softmax policy for the given set
        '''
        N = 2 * d**2 * np.log(d)
        gamma = S.shape[0]

        U = lamda * N * gamma * np.identity(d)
        for X in S:
            g_opt_design = g_optimal_design(X)
            U += N/2 * information_matrix_set(X , g_opt_design)

        n = 1
        tau_lengths = []
        W_collection = []
        current_tau = []
        current_W = U

        for t in tqdm(range(N * gamma)):
            current_tau.append(t)
            x = sample_softmax(S[int(t%gamma)] , np.linalg.inv(current_W) , np.log(len(S[t])))
            U += np.outer(x , x)
            if np.linalg.det(U) > 2 * np.linalg.det(current_W):
                n += 1
                tau_lengths.append(len(current_tau))
                current_tau = []
                W_collection.append(np.linalg.inv(current_W))
                current_W = U

        for i in range(len(tau_lengths)):
            if tau_lengths[i] < gamma:
                tau_lengths[i] = 0
            W_collection[i] *= N*gamma
            
        p = np.array(tau_lengths) / np.sum(tau_lengths)

        p_non_zero = []
        W_collection_final = []
        for i in range(len(tau_lengths)):
            if tau_lengths[i] != 0:
                p_non_zero.append(p[i])
                W_collection_final.append(W_collection[i])

        return p_non_zero , W_collection_final

    def sample_mixed_softmax(self , X):
        '''
        samples an arm from the mixed softmax policy
        w/p 1/2 samples from G optimal design
        else samples from softmax policy
        '''
        if np.random.choice([0,1]) == 0:
            g_opt_design = g_optimal_design(X)
            return X[np.random.choice(len(X) , p = g_opt_design)]
        else:
            index = np.random.choice(len(self.p_array) , p = self.p_array)
            return sample_softmax(X , self.matrices[index] , np.log(len(self.arms)))
        
    def sample_G_optimal(self , X):
        '''
        samples an arm from the G optimal design
        '''
        g_opt_design = g_optimal_design(X)
        return X[np.random.choice(len(X) , p = g_opt_design)]