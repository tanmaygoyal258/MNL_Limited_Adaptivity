import numpy as np
from tqdm import tqdm
from utils import gradient_MNL , prob_vector , weighted_norm , minimize_omd_loss
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from time import time

class MLogB():

    def __init__(self , params , oracle , arm_set , reward_vec):

        self.arm_set = arm_set
        self.oracle = oracle
        self.reward_vec = reward_vec
        
        self.dim_arms = params["dim_arms"]
        self.reward_vec_norm_ub = params["reward_vec_norm_ub"]
        self.param_norm_ub = params["param_norm_ub"]
        self.num_contexts = params["num_contexts"]
        self.horizon = params["horizon"]
        self.num_outcomes = params["num_outcomes"]

        self.alpha = 2*(1 + self.param_norm_ub) + np.log(self.num_outcomes+1)
        self.lamda = np.sqrt(self.num_outcomes) * self.alpha * self.param_norm_ub
        self.eta = self.param_norm_ub/2 + np.log(self.num_outcomes+1)/4

        self.H = self.lamda * np.identity(self.dim_arms*self.num_outcomes)
        self.theta = np.zeros((self.dim_arms*self.num_outcomes,1))

        self.regret_arr = []
        self.time_arr = []

        self.outfile = open(params["data_path"] + "/outfile.txt" , 'w')
    
    def play_arm(self , arms , t):
        beta = np.log(self.num_outcomes) * np.log(t+1) * np.sqrt(self.num_outcomes * self.dim_arms)
        # Find max UCB arm
        ucb_vals = [self.calculate_UCB(arm , beta) for arm in arms]
        return arms[np.argmax(ucb_vals)]
    
    def update(self , outcome , played_arm):
        H_tilde = self.H.copy() + self.eta * np.kron(gradient_MNL(played_arm , self.theta , self.dim_arms , self.num_outcomes) , np.outer(played_arm , played_arm))

        outcome_vector = [0 for _ in range(self.num_outcomes)]
        if outcome != 0:
            outcome_vector[outcome-1] = 1
        outcome_vector = np.array(outcome_vector).reshape(1 , self.num_outcomes)
        loss_gradient = np.kron((outcome_vector - prob_vector(played_arm , self.theta , self.dim_arms , self.num_outcomes)) , played_arm.T) + self.lamda * self.theta.T

        # Z = self.theta - self.eta * np.linalg.inv(H_tilde) @ loss_gradient.T

        # # find eiegevector corresponding to minimum eigenvalue
        # eigenval , eigenvec = np.linalg.eig(H_tilde)
        # combination = [(eigenval[i] , eigenvec[:,i]) for i in range(len(eigenval))]
        # combination.sort(reverse = False , key = lambda x: x[0])
        # min_eigvec = combination[0][1].reshape(-1,1)
        # self.theta += min_eigvec
        # self.theta /= np.linalg.norm(self.theta)
        # self.theta *= self.param_norm_ub

        constraint = [{'type' : "ineq" , "fun": lambda x: self.param_norm_ub**2 - np.dot(x,x)}]        
        self.theta = minimize(minimize_omd_loss , self.theta.reshape(-1,) , args = (self.theta.reshape(-1,) , loss_gradient , self.eta , H_tilde) , constraints = constraint , tol = 0.25*self.num_outcomes).x
        self.outfile.write(f"{self.theta}")

        self.H += np.kron(gradient_MNL(played_arm , self.theta , self.dim_arms , self.num_outcomes) , np.outer(played_arm , played_arm))

        # constraint = [{'type' : "eq" , "fun": lambda x: self.S-1-np.linalg.norm(x)}]        
        # self.theta = minimize(minimize_omd_loss , self.theta , args = (self.theta , loss_gradient , self.eta , H_tilde) , constraints = constraint)


    def play_algorithm(self):
        for t in tqdm(range(self.horizon)):
            arms = self.arm_set[t] if self.num_contexts == self.horizon else self.arm_set[np.random.choice(self.num_contexts)]
            
            pull_start = time()
            played_arm = self.play_arm(arms , t)
            pull_end = time()

            best_arm , best_arm_reward = self.find_best_arm_reward(arms)
            self.regret_arr.append(best_arm_reward - self.oracle.expected_reward(played_arm))
            outcome = self.oracle.pull(played_arm)

            update_start = time()
            self.update(outcome , played_arm)
            update_end = time()

            self.time_arr.append(update_end + pull_end - pull_start - update_start)
        return self.regret_arr , self.time_arr

    def find_best_arm_reward(self , arm_set):
        '''
        finds the best arm with best expected rewards
        '''
        arm_rewards = [self.oracle.expected_reward(arm) for arm in arm_set]
        best_arm_index = np.argmax(arm_rewards)
        return arm_set[best_arm_index] , arm_rewards[best_arm_index]
            

    def calculate_UCB(self , arm , beta):
        # print(prob_vector(arm , self.theta , self.dim_arms , self.num_outcomes).shape)
        return np.dot(self.reward_vec , prob_vector(arm , self.theta , self.dim_arms , self.num_outcomes).reshape(self.num_outcomes,)) + self.error1(arm,beta) + self.error2(arm,beta)
    
    def error1(self , arm , beta):
        return beta * np.sqrt(np.dot(self.reward_vec , np.kron(gradient_MNL(arm , self.theta, self.dim_arms , self.num_outcomes) , arm.T) @ np.linalg.inv(self.H) @ np.kron(gradient_MNL(arm , self.theta, self.dim_arms , self.num_outcomes) , arm) @ self.reward_vec))
    
    def error2(self , arm , beta):
        eigenval , eigenvec = np.linalg.eig(np.kron(np.identity(self.num_outcomes) , np.outer(arm,arm)) @ np.linalg.inv(self.H))
        combination = [(eigenval[i] , eigenvec[:,i]) for i in range(len(eigenval))]
        combination.sort(reverse = True , key = lambda x: x[0])
        max_eigenval = combination[0][0]
        return 3 * self.reward_vec_norm_ub * beta * np.sqrt(max_eigenval)