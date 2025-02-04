import numpy as np
from tqdm import tqdm
from CoreIdentification import CoreIdentification
from utils import g_optimal_design , information_matrix_set , sample_softmax , weighted_norm 
from scipy.linalg import sqrtm

class Distributional_G_optimal():
    def __init__(self , optimal_design_alg ,  BS_constant):

        self.p_array = None
        self.matrices = None
        self.opt_design_alg = optimal_design_alg
        self.BS_constant = BS_constant

    def update_parameters(self ,  lamda  , multiset , dim_arms):
        '''
        finds the probability and matrices for the mixed softmax policy(ies)
        '''
        
        c = 6
        core_set = CoreIdentification(lamda , c , multiset , dim_arms).findCoreSet()
        self.p_array , self.matrices =  self.obtainMixedSoftmaxParameters(core_set , lamda , dim_arms)
    
    def obtainMixedSoftmaxParameters(self , multiset , lamda , d):
        '''
        returns the mixed softmax policy for the given set
        '''
        N = 2 * d**2 * np.log(d)
        gamma = len(multiset)

        U = lamda * N * gamma * np.identity(d)
        print("*"*100)
        print(lamda , N , gamma , np.linalg.det(U))
        for _ , X in enumerate(multiset):
            g_opt_design = g_optimal_design(X , algorithm = self.opt_design_alg , BS = self.BS_constant)
            print(information_matrix_set(g_opt_design , X) , np.linalg.det(N/2*information_matrix_set(g_opt_design , X)))
            U += N/2 * information_matrix_set(g_opt_design , X)
            print(np.linalg.det(U))

        n = 1
        tau_lengths = []
        W_collection = []
        current_tau = []
        current_W = U.copy()
        print("Starting U det is " , np.linalg.det(current_W))

        for t in tqdm(range(int(N) * gamma)):
            current_tau.append(t)
            x = sample_softmax(multiset[int(t%gamma)] , np.linalg.inv(current_W) , np.log(len(multiset[int(t%gamma)] )))[0][2]
            print("Trace ", np.trace(np.outer(x,x)))
            U += np.outer(x , x)
            print(np.linalg.det(U) , np.linalg.det(current_W))
            if np.linalg.det(U) > 2 * np.linalg.det(current_W):
                n += 1
                tau_lengths.append(len(current_tau))
                current_tau = []
                W_collection.append(np.linalg.inv(current_W))
                current_W = U.copy()


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
        else samples from ith mixed softmax policy
        '''
        if np.random.choice([0,1]) == 0:
            return "G_optimal" , self.sample_G_optimal(X)
        else:
            index = np.random.choice(len(self.p_array) , p = self.p_array)
            return "softmax" ,  sample_softmax(X , self.matrices[index] , np.log(len(X)))
        
    def sample_G_optimal(self ,  X):
        '''
        samples an arm from the G optimal design
        '''
        g_opt_design = g_optimal_design(X , algorithm = self.opt_design_alg , BS = self.BS_constant)
        g_opt_design /= np.sum(g_opt_design)
        # information_matrix = information_matrix_set(g_opt_design , X)
        # print([weighted_norm(a , np.linalg.inv(information_matrix)) for a in X])
        arm_index = np.random.choice(len(X) , p = g_opt_design)
        return g_opt_design , arm_index , X[arm_index]
        