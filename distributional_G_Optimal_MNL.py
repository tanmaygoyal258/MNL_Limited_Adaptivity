import numpy as np
from tqdm import tqdm
from CoreIdentification import CoreIdentification
from utils import g_optimal_design , information_matrix_set , sample_softmax , gradient_MNL
from scipy.linalg import sqrtm

class Distributional_G_optimal():
    def __init__(self , optimal_design_alg ,  BS_constant):

        self.p_array = []
        self.matrices = []
        self.opt_design_alg = optimal_design_alg
        self.BS_constant = BS_constant

    def update_parameters(self ,  lamda , num_outcomes , multiset , dim_arms , theta , normalizing_func):
        '''
        finds the probability and matrices for the mixed softmax policy(ies)
        '''
        self.p_array = []
        self.matrices = []
        c = 6

        for i in range(num_outcomes):
            scaled_ith_set = self.scale(multiset , i , theta , normalizing_func)
            core_set = CoreIdentification(lamda , c , scaled_ith_set , dim_arms * num_outcomes).findCoreSet()
            p , m =  self.obtainMixedSoftmaxParameters(core_set , lamda , dim_arms * num_outcomes)
            self.p_array.append(p)
            self.matrices.append(m)

    def scale(self , multiset , i , theta , normalizing_func):
        '''
        scales the arms by the ith column of the gradient
        '''
        scaled_arm_set = []
        for X in multiset:
            temp_arms = [np.kron(sqrtm(gradient_MNL(x  ,theta))[:,i] , x) / normalizing_func(x , theta) for x in X]
            scaled_arm_set.append([x/np.linalg.norm(x) for x in temp_arms])
            # scaled_arm_set.append(temp_arms)
        return scaled_arm_set
    
    def obtainMixedSoftmaxParameters(self , multiset , lamda , d):
        '''
        returns the mixed softmax policy for the given set
        '''
        N = 2 * d**2 * np.log(d)
        gamma = len(multiset)

        U = lamda * N * gamma * np.identity(d)
        for X in multiset:
            g_opt_design = g_optimal_design(X , algorithm = self.opt_design_alg , BS = self.BS_constant)
            U += N/2 * information_matrix_set(g_opt_design , X)

        n = 1
        tau_lengths = []
        W_collection = []
        current_tau = []
        current_W = U.copy()

        for t in tqdm(range(int(N) * gamma)):
            current_tau.append(t)
            x = sample_softmax(multiset[int(t%gamma)] , np.linalg.inv(current_W) , np.log(len(multiset[int(t%gamma)] )))
            U += np.outer(x , x)
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

    def sample_mixed_softmax(self , X , i):
        '''
        samples an arm from the mixed softmax policy
        w/p 1/2 samples from G optimal design
        else samples from ith mixed softmax policy
        '''
        if np.random.choice([0,1]) == 0:
            return self.sample_G_optimal(X)
        else:
            index = np.random.choice(len(self.p_array[i]) , p = self.p_array[i])
            return sample_softmax(X , self.matrices[i][index] , np.log(len(X)))
        
    def sample_G_optimal(self ,  X):
        '''
        samples an arm from the G optimal design
        '''
        g_opt_design = g_optimal_design(X , algorithm = self.opt_design_alg , BS = self.BS_constant)
        return X[np.random.choice(len(X) , p = g_opt_design)]
    
    def sample_MNL_policy(self , X , num_outcomes , theta , normalizing_func):
        '''
        samples an arm from G_optimal design with probability 1/K+1
        otherwise samples from mixed softmax policies over the scaled arms
        '''
        choice = np.random.choice([_ for _ in range(num_outcomes + 1)])
        if choice == 0:
            return self.sample_G_optimal(X)
        else:
            scaled_set = self.scale([X] , choice-1 , theta , normalizing_func)[0]
            scaled_mapping = {tuple(scaled) : normal for scaled , normal in zip(scaled_set , X)}
            scaled_item_chosen = self.sample_mixed_softmax(scaled_set , choice-1)
            return scaled_mapping[tuple(scaled_item_chosen)]
