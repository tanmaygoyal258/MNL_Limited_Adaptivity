import numpy as np
from utils import *
from tqdm import tqdm
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from time import time

class RS_MNL:
    def __init__(self, params , oracle):
        # setting the reward vector and oracle
        self.reward_vec = params["reward_vec"]    
        self.oracle = oracle
        
        # setting the variables from param
        self.dim_arms = params["dim_arms"]
        self.num_arms = params["num_arms"]
        self.num_contexts = params["num_contexts"]
        self.num_outcomes = params["num_outcomes"]
        self.reward_norm_ub = params["reward_norm_ub"]
        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.delta= params["failure_level"]
        self.number_arms = params["num_arms"]
        self.theta_star = params["theta_star"]

        # initializing the arm set
        if self.num_contexts != self.horizon:
            self.arm_set = self.create_arm_set(np.random.default_rng(params["arm_seed"]))
        self.arm_rng = np.random.default_rng(params["arm_seed"])

        # might have tune variables these for varying S
        self.doubling_constant = 1 
        self.lmbda = self.num_outcomes * self.dim_arms * np.log(self.horizon) / np.sqrt(self.param_norm_ub)
        self.gamma = self.reward_norm_ub * self.param_norm_ub* np.sqrt(self.dim_arms*self.num_outcomes*np.log(self.horizon/self.delta))
        self.e = np.exp(1.0)
        
        # initializing matrices
        self.V_inv = (1.0 / self.lmbda) * np.eye(self.dim_arms*self.num_outcomes)
        self.curr_H = self.lmbda * np.eye(self.dim_arms*self.num_outcomes)
        self.prev_H = 0 * np.eye(self.dim_arms*self.num_outcomes)
        self.theta_hat_tau = np.zeros((self.dim_arms*self.num_outcomes,))
        self.X, self.Y = [], []
        self.warmup_flag = False
        
        # initializing arrays
        self.regret_arr = []
        self.time_arr = []
        self.switches_arr = []
        self.hits = 0

    def play_arm(self , arms, t):
        """
        chooses an arm out of {arms} to play
        """
        # Check warm up
        if np.linalg.det(self.curr_H) >= ((1 + self.doubling_constant) * np.linalg.det(self.prev_H)):
            self.hits += 1
            theta_received , succ_flag = minimize_loss_torch(self.theta_hat_tau , self.X , self.Y , self.lmbda , self.dim_arms)
            if succ_flag is True:
                self.theta_hat_tau = theta_received
        
            self.curr_H = self.lmbda * np.eye(self.dim_arms*self.num_outcomes)
            for arm in self.X:
                self.curr_H += np.kron(gradient_MNL(arm , self.theta_hat_tau) , np.outer(arm, arm))
            self.prev_H = self.curr_H.copy()
        
        self.switches_arr.append(self.hits)
        # Find max UCB arm
        ucb_vals = [self.calculate_UCB(arm) for arm in arms]
        return arms[np.argmax(ucb_vals)]
    
    def update(self , outcome , played_arm):
        """
        updates the parameters after the batch
        """
        self.X.append(played_arm)
        self.Y.append(outcome)

        # Update current H matrix
        self.curr_H += np.kron(gradient_MNL(played_arm , self.theta_hat_tau) , np.outer(played_arm, played_arm))
            
    def play_algorithm(self):
        """
        plays the algorithm in a standard format: choose an arm, obtain reward, update params
        """
        
        for t in tqdm(range(self.horizon)):
            # obtain the arms
            if self.num_contexts != self.horizon:
                arms = self.slot_arms[np.random.choice(self.num_contexts)]
            else:
                arms = self.create_arm_set(self.arm_rng)
            
            # find the arm to play
            played_arm = self.play_arm(arms, t)
            
            # obtain the reward and regret
            best_arm , best_arm_reward = self.find_best_arm_reward(arms)
            self.regret_arr.append(best_arm_reward - self.oracle.expected_reward(played_arm))
            outcome = self.oracle.pull(played_arm)

            # update parameters
            self.update(outcome , played_arm)
            
        print(f"Switching Criteria was hit {self.hits} times")
        return self.regret_arr , self.switches_arr

    def find_best_arm_reward(self , arm_set):
        '''
        finds the best arm with best expected rewards
        '''
        arm_rewards = [self.oracle.expected_reward(arm) for arm in arm_set]
        best_arm_index = np.argmax(arm_rewards)
        return arm_set[best_arm_index] , arm_rewards[best_arm_index]
            

    def calculate_UCB(self , arm):
        """
        Calculates the Upper Confidence Bound(UCB) for an arm
        """

        def error1(arm):
            return np.sqrt(1+self.doubling_constant) * self.gamma * np.linalg.norm(sqrtm(np.linalg.inv(self.curr_H)) @ np.kron(np.eye(self.num_outcomes) , arm.reshape(-1,1)) @ gradient_MNL(arm , self.theta_hat_tau) @ self.reward_vec , ord = 2)

        def error2(arm):
            return 3 * (1 + self.doubling_constant) * self.reward_norm_ub * self.gamma**2 * np.linalg.norm(np.kron(np.eye(self.num_outcomes) , arm.reshape(-1,1).T) @ sqrtm(np.linalg.inv(self.curr_H)), ord=2)**2

        return np.dot(self.reward_vec , prob_vector(arm , self.theta_hat_tau)[1:]) + error1(arm) + error2(arm)

    def create_arm_set(self , arm_rng):
        """
        creates an arm set using a random generator
        """
        arms = []
        for a in range(self.number_arms):
            arm = [arm_rng.random()*2 - 1 for i in range(self.dim_arms)]
            arm = arm / np.linalg.norm(arm)
            arms.append(arm)
        return arms