import numpy as np
from distributional_G_Optimal_MNL import Distributional_G_optimal
from B_MNL import B_MNL
from MLogB import MLogB

class MNLOracle():
    def __init__(self , theta_star , reward_vec):
        self.theta_star = theta_star
        self.reward_vec = reward_vec
    
    def prob_vec(self , arm):
        '''
        returns a softmax probability vector with an additional 1 (=exp(0)) in the denominator to account for no action chosen
        '''
        self.theta_star_temp = np.reshape(self.theta_star , (-1 , arm.shape[0]))
        inner_products = self.theta_star_temp @ arm
        # adding the 0 for no action chosen
        inner_products = np.hstack([[0] , inner_products])
        return np.exp(inner_products) / np.sum(np.exp(inner_products))

    def expected_reward(self , arm):
        '''
        the expected reward is the inner product between the reward vec and the probability vector
        '''
        # we exclude the first element which is probability of no action with zero reward
        prob_vec = self.prob_vec(arm)[1:]
        return np.dot(prob_vec , self.reward_vec)

    def pull(self , arm):
        '''
        the outcome is sampled from the probability vector and the reward returned is from the reward vector
        Note that the reward for outcome zero is zero which is not reflected in the reward vector
        '''
        index = np.random.choice(len(self.prob_vec(arm)) , p = self.prob_vec(arm))
        if index != 0:
            return index , self.reward_vec[index-1]
        else:
            return 0 , 0
        
class MNLEnv_Batched():
    def __init__(self , params , arms , theta_star , reward_vec):
        
        self.seed = params["seed"]

        self.oracle = MNLOracle(theta_star , reward_vec)

        self.horizon = params["horizon"]
        self.num_batches = params["num_batches"]
        self.dim_arms = params["dim_arms"]
        self.param_norm_ub = params["param_norm_ub"]
        self.reward_vec_norm_ub = params["reward_vec_norm_ub"]
        
        self.optimal_design_alg = params["optimal_design_alg"]
        self.bs_constant = params["BS_constant"]

        # self.regret_arr = np.empty(self.horizon)
        # self.reward_arr = np.empty(self.horizon)
        # self.update_time_arr = np.empty(self.horizon)
        # self.pulling_time_arr = np.empty(self.horizon)
        # self.total_time_arr = np.empty(self.horizon)
        
        self.arms = arms

        self.g_distributional_design = Distributional_G_optimal(self.optimal_design_alg , self.bs_constant)
        self.batch_endpoints = self.get_batch_endpoints()

        # create an instance of the algorithm    
        if params["alg_name"] == "B_MNL": 
            self.algorithm = B_MNL(params , self.oracle , self.arms , self.batch_endpoints , self.g_distributional_design)
        self.regret_arr = self.algorithm.play_algorithm()
        

    def get_batch_endpoints(self):
        '''
        returns the lengths of each batch
        '''
        log_log_T = np.log(np.log(self.horizon))
        M = self.num_batches
        alpha = self.horizon ** (2**(M-1)/(2**M - 2)) if M <= log_log_T else 2*np.sqrt(self.horizon)

        batch_lengths = [int(alpha)]
        while (np.sum(batch_lengths.copy()) < self.horizon):
            batch_lengths.append(int(alpha * np.sqrt(batch_lengths[-1])))
        
        batch_endpoints = np.cumsum(batch_lengths)
        batch_endpoints = np.hstack([[0] , batch_endpoints])
        batch_endpoints = np.clip(batch_endpoints , 0 , self.horizon)
        assert batch_endpoints[-1] == self.horizon
        return batch_endpoints
