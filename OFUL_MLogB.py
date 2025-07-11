import numpy as np
from tqdm import tqdm
# from utils_backup import gradient_MNL , prob_vector , weighted_norm , minimize_omd_loss
from utils import *
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from time import time

class MLogB():

    def __init__(self , params , oracle):

        self.reward_vec = params["reward_vec"]
        self.oracle = oracle
        
        # setting variables from params
        self.dim_arms = params["dim_arms"]
        self.reward_norm_ub = params["reward_norm_ub"]
        self.param_norm_ub = params["param_norm_ub"]
        self.num_contexts = params["num_contexts"]
        self.horizon = params["horizon"]
        self.num_outcomes = params["num_outcomes"]
        self.number_arms = params["num_arms"]

        self.alpha = 2*(1 + self.param_norm_ub) + np.log(self.num_outcomes+1)
        self.lamda = np.sqrt(self.num_outcomes) * self.alpha * self.param_norm_ub
        self.eta = self.param_norm_ub/2 + np.log(self.num_outcomes+1)/4

        self.H = self.lamda * np.identity(self.dim_arms*self.num_outcomes)
        self.theta = np.zeros(self.dim_arms*self.num_outcomes)

        self.regret_arr = []
        self.time_arr = []

        if self.num_contexts != self.horizon:
            self.arm_set = self.create_arm_set(np.random.default_rng(params["arm_seed"]))
        self.arm_rng = np.random.default_rng(params["arm_seed"])

    def play_arm(self , arms , t):
        """
        chooses an arm out of {arms} to play
        """
        beta = np.log(self.num_outcomes) * np.log(t+1) * np.sqrt(self.num_outcomes * self.dim_arms)
        # Find max UCB arm
        ucb_vals = [self.calculate_UCB(arm , beta) for arm in arms]
        return arms[np.argmax(ucb_vals)]
    
    def update(self , outcome , played_arm):
        """
        updates the parameters after the batch
        """
        H_tilde = self.H.copy() + self.eta * np.kron(gradient_MNL(played_arm , self.theta) , np.outer(played_arm , played_arm))

        outcome_vector = [0 for _ in range(self.num_outcomes)]
        if outcome != 0:
            outcome_vector[outcome-1] = 1
        outcome_vector = np.array(outcome_vector)
        loss_gradient = np.kron(prob_vector(played_arm , self.theta)[1:] - outcome_vector , played_arm)

        Z = self.theta - self.eta * np.linalg.inv(H_tilde) @ loss_gradient.T

        # find eigenvector corresponding to minimum eigenvalue
        eigenval , eigenvec = np.linalg.eig(H_tilde)
        combination = [(eigenval[i] , eigenvec[:,i]) for i in range(len(eigenval))]
        combination.sort(reverse = False , key = lambda x: x[0])
        min_eigvec = combination[0][1].reshape(-1,)
        self.theta = min_eigvec + Z
        self.theta /= (np.linalg.norm(self.theta) / self.param_norm_ub)

        self.H += np.kron(gradient_MNL(played_arm , self.theta) , np.outer(played_arm , played_arm))


    def play_algorithm(self):
        for t in tqdm(range(self.horizon)):
            """
            plays the algorithm in a standard format: choose an arm, obtain reward, update params
            """
            # obtain the arms
            if self.num_contexts != self.horizon:
                arms = self.slot_arms[np.random.choice(self.num_contexts)]
            else:
                arms = self.create_arm_set(self.arm_rng)
            
            # select the arm to play
            played_arm = self.play_arm(arms , t)

            # find the reward and regret
            best_arm , best_arm_reward = self.find_best_arm_reward(arms)
            self.regret_arr.append(best_arm_reward - self.oracle.expected_reward(played_arm))
            outcome = self.oracle.pull(played_arm)

            # update parameters
            self.update(outcome , played_arm)

        return self.regret_arr

    def find_best_arm_reward(self , arm_set):
        '''
        finds the best arm with best expected rewards
        '''
        arm_rewards = [self.oracle.expected_reward(arm) for arm in arm_set]
        best_arm_index = np.argmax(arm_rewards)
        return arm_set[best_arm_index] , arm_rewards[best_arm_index]
            

    def calculate_UCB(self , arm , beta):
        def error1(arm , beta):
            return beta * np.linalg.norm(sqrtm(np.linalg.inv(self.H)) @ np.kron(np.eye(self.num_outcomes) , arm.reshape(-1,1)) @ gradient_MNL(arm , self.theta) @ self.reward_vec , ord = 2)
    
        def error2(arm , beta):
            return 3 * self.reward_norm_ub * beta**2 * np.linalg.norm(np.kron(np.eye(self.num_outcomes) , arm.reshape(-1,1).T) @ sqrtm(np.linalg.inv(self.H)), ord=2)**2

        return np.dot(self.reward_vec , prob_vector(arm , self.theta)[1:]) + error1(arm , beta) + error2(arm , beta)
    
    

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