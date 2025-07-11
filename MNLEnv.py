import numpy as np
from OFUL_MLogB import MLogB
from RS_GLincb import RS_GLinUCB
from RS_MNL import RS_MNL
from ada_OFU_ECOLog import ada_OFU_ECOLog
from ofulogplus import OFULogPlus
from utils import *

class MNLOracle():
    '''
    Simulates a Multinomial Logistic Oracle
    '''
    def __init__(self , reward_rng , theta_star , reward_vec):
        self.reward_rng = reward_rng
        self.theta_star = theta_star
        self.reward_vec = reward_vec
        self.num_outcomes = len(reward_vec)
        self.dimension = int(len(theta_star)/self.num_outcomes)
        self.theta_star_with_zero = np.hstack(([0 for _ in range(self.dimension)] , self.theta_star))
        self.reward_vec_with_zero = np.hstack(([0], self.reward_vec))

    def prob_vec(self , arm):
        '''
        returns a softmax probability vector with an additional 1 (=exp(0)) in the denominator to account for no action chosen
        '''
        inner_products = np.kron(np.eye(self.num_outcomes+1) , arm) @ self.theta_star_with_zero
        return np.exp(inner_products) / np.sum(np.exp(inner_products))

    def expected_reward(self , arm):
        '''
        the expected reward is the inner product between the reward vec and the probability vector
        '''
        return np.dot(self.prob_vec(arm) , self.reward_vec_with_zero)

    def pull(self , arm):
        '''
        the outcome is sampled with probability {prob_vector}
        '''
        return self.reward_rng.choice(len(self.prob_vec(arm)) , p = self.prob_vec(arm))

class MNLEnv():

    def __init__(self , params):
        
        self.alg_name = params["alg_name"]
        self.oracle = MNLOracle(np.random.default_rng(params["reward_seed"]) , params["theta_star"] , params["reward_vec"])
        self.number_arms = params["num_arms"]
        self.dim = params["dim_arms"]
        
        # create an instance of the algorithm    
        if self.alg_name.lower() == "rs_mnl": 
            self.algorithm = RS_MNL(params ,self.oracle)
        elif self.alg_name.lower() == "mlogb":
            self.algorithm = MLogB(params , self.oracle)
        elif self.alg_name.lower() == "ada_ofu_ecolog":
            self.algorithm = ada_OFU_ECOLog(params , self.oracle)
        elif self.alg_name.lower() == "ofulogplus":
            self.algorithm = OFULogPlus(params , self.oracle)
        elif self.alg_name.lower() == "rs_glincb":
            self.algorithm = RS_GLinUCB(params , self.get_kappa_sigmoid(params) , self.oracle)

        # run the algorithm and obtain results
        results = self.algorithm.play_algorithm()
        try:
            self.regret_arr , self.switch_arr = results
        except:
            # there is no switching array
            self.regret_arr = results

    def get_kappa_sigmoid(self , params):
        """
        finds kappa (with repsect to theta_star)
        """
        # sets the random generator for the arms
        arm_rng = np.random.default_rng(params["arm_seed"])
        theta = params["theta_star"]
        kappa = -np.inf
        for _ in range(params["num_contexts"]):
            arms = self.create_arm_set(arm_rng)
            mu_dot = [dsigmoid(np.dot(theta , arm)) for arm in arms]
            kappa = max(kappa , 1.0/np.min(mu_dot))
        return kappa

    def create_arm_set(self , arm_rng):
        """
        creates an arm set using a random generator
        """
        arms = []
        for a in range(self.number_arms):
            arm = [arm_rng.random()*2 - 1 for i in range(self.dim)]
            arm = arm / np.linalg.norm(arm)
            arms.append(arm)
        return arms