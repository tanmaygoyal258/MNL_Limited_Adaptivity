import numpy as np
from distributional_G_Optimal import Distributional_G_optimal

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
        prob_vec = self.prob_vec(arm)
        return np.dot(prob_vec , self.reward_vec)

    def pull(self , arm):
        '''
        the outcome is sampled from the probability vector and the reward returned is from the reward vectopr
        '''
        return self.reward_vec[np.random.choice(len(self.reward_vec) , p = self.prob_vec(arm))]
        
class MNLEnv():

    def __init__(self , params , arms , theta_star , reward_vec):
        
        self.params = params
        self.seed = params["seed"]

        self.oracle = MNLOracle(theta_star , reward_vec)

        self.horizon = params["horizon"]
        self.dim_arms = params["dim_arms"]
        self.param_norm_ub = params["param_norm_ub"]
        self.reward_vec_norm_ub = params["reward_vec_norm_ub"]
        
        self.optimal_design_alg = params["optimal_design_alg"]
        self.bs_constant = params["BS_constant"]

        self.regret_arr = np.empty(self.horizon)
        self.reward_arr = np.empty(self.horizon)
        # self.update_time_arr = np.empty(self.horizon)
        # self.pulling_time_arr = np.empty(self.horizon)
        # self.total_time_arr = np.empty(self.horizon)
        
        self.arms = arms
        self.ctr = 0

        self.g_distributional_design = Distributional_G_optimal()

        np.random.seed(self.seed)
        