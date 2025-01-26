import numpy as np
from distributional_G_Optimal_Linear import Distributional_G_optimal
from BatchLinUCB import BatchLinUCB
from utils import weighted_norm

class LinearOracle():
    def __init__(self , theta_star):
        self.theta_star = theta_star

    def expected_reward(self , arm):
        '''
        the expected reward is the inner product between the arm and thetastar
        '''
        return np.dot(arm , self.theta_star)

    def pull(self , arm):
        '''
        the reward is the inner product with some gaussian noise
        '''
        return np.dot(arm , self.theta_star) + np.random.normal(0 , 0.01)
        
class LinearEnv():
    def __init__(self , params , arms , theta_star):
        
        self.seed = params["seed"]

        self.oracle = LinearOracle(theta_star)

        self.horizon = params["horizon"]
        self.num_batches = params["num_batches"]
        self.dim_arms = params["dim_arms"]
        self.param_norm_ub = params["param_norm_ub"]
        
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
        self.algorithm = BatchLinUCB(params , self.oracle , self.arms , self.batch_endpoints , self.g_distributional_design)
        self.regret_arr = self.algorithm.play_algorithm()
        

    def get_batch_endpoints(self):
        '''
        returns the lengths of each batch
        '''
        M = int(np.ceil(np.log(np.log(self.horizon)))) + 1

        batch_lengths = []
        for m in range(M):
            if m == 0:
                batch_lengths.append(int(np.sqrt(self.horizon)))
            elif m == 1:
                batch_lengths.append(int(2*np.sqrt(self.horizon)))
            elif m == M-1:
                batch_lengths.append(self.horizon)
            else:
                batch_lengths.append(int(self.horizon ** (1-2**(-m))))

        batch_endpoints = np.cumsum(batch_lengths)
        batch_endpoints = np.hstack([[0] , batch_endpoints])
        batch_endpoints = np.clip(batch_endpoints , 0 , self.horizon)
        assert batch_endpoints[-1] == self.horizon
        return batch_endpoints
