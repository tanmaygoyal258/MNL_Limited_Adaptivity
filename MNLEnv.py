import numpy as np
from distributional_G_Optimal_MNL import Distributional_G_optimal
from B_MNL import B_MNL
from MLogB import MLogB
from RS_GLincb import RS_GLinUCB
from RS_MNL import RS_MNL
from ofulogplus import OFULogPlus
from utils import dsigmoid , sigmoid
from ada_OFU_ECOLog import ada_OFU_ECOLog

class MNLOracle():
    def __init__(self , theta_star , reward_vec):
        self.theta_star = theta_star
        self.reward_vec = reward_vec
    
    def prob_vec(self , arm):
        '''
        returns a softmax probability vector with an additional 1 (=exp(0)) in the denominator to account for no action chosen
        '''
        self.theta_star_temp = np.reshape(self.theta_star , (-1 , len(arm)))
        inner_products = self.theta_star_temp @ arm
        inner_products = inner_products.reshape(-1,)
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
        outcome = np.random.choice(len(self.prob_vec(arm)) , p = self.prob_vec(arm))
        return outcome

    # def expected_reward(self , arm):
    #     '''
    #     the expected reward is the sigmoid of the inner product between arm and optimal param
    #     '''
    #     return sigmoid(np.dot(arm , self.theta_star))

    # def expected_reward_linear(self , arm):
    #     '''
    #     the linear expected reward is the inner product between arm and optimal param
    #     '''
    #     return (np.dot(arm , self.theta_star))

    # def pull(self , arm):
    #     '''
    #     the actual reward is sampled from a Bernoulli Distribution with mean equal to the expected reward
    #     '''
    #     reward = int(np.random.rand() < self.expected_reward(arm))
    #     return reward,reward

class MNLEnv():

    def __init__(self , params , arms , theta_star , reward_vec):
        
        self.seed = params["seed"]
        self.alg_name = params["alg_name"]
        self.oracle = MNLOracle(theta_star , reward_vec)

        self.horizon = params["horizon"]
        self.dim_arms = params["dim_arms"]
        self.param_norm_ub = params["param_norm_ub"]
        self.reward_vec_norm_ub = params["reward_vec_norm_ub"]
    
        # self.regret_arr = np.empty(self.horizon)
        # self.reward_arr = np.empty(self.horizon)
        # self.update_time_arr = np.empty(self.horizon)
        # self.pulling_time_arr = np.empty(self.horizon)
        # self.total_time_arr = np.empty(self.horizon)
        
        self.arms = arms

        # create an instance of the algorithm    
        # self.algorithm = RS_MNL(params , arms , self.get_kappa(arms , theta_star) , self.oracle)
        if self.alg_name == "rs_mnl": 
            self.algorithm = RS_MNL(params , arms , reward_vec ,self.oracle)
        elif self.alg_name == "mlogb":
            self.algorithm = MLogB(params , self.oracle , arms , reward_vec)
        elif self.alg_name == "ada_ofu_ecolog":
            self.algorithm = ada_OFU_ECOLog(params , arms , self.oracle)
        elif self.alg_name == "ofulogplus":
            self.algorithm = OFULogPlus(params , arms , self.oracle)
        elif self.alg_name == "rs_glincb":
            self.algorithm = RS_GLinUCB(params , arms , self.get_kappa(arms , theta_star) , self.oracle)
        # self.algorithm = ada_OFU_ECOLog(params , arms, self.oracle)
        # if params["alg_name"] == "B_MNL": 
        #     self.algorithm = B_MNL(params , self.oracle , self.arms , self.batch_endpoints , self.g_distributional_design)
        # elif params["alg_name"] == "MLogB":
        #     self.algorithm = MLogB(params , self.oracle , self.arms)

        results = self.algorithm.play_algorithm()
        try:
            self.regret_arr , self.time_arr , self.switch_arr = results
        except:
            # not a switching array
            self.regret_arr , self.time_arr = results

    def get_kappa(self , arm_set , thetastar):
        all_arms = []
        for arms in arm_set:
            all_arms += arms.copy()
        min_mu_dot = np.inf
        for arm in all_arms:
            min_mu_dot = min(dsigmoid(np.dot(thetastar , arm)) , min_mu_dot)
        return 1.0/min_mu_dot