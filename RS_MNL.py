import numpy as np
from utils import mat_norm, solve_glm_mle, gradient_MNL , prob_vector , log_loss
from tqdm import tqdm
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from time import time

class RS_MNL:
    def __init__(self, params , arm_set, reward_vec , oracle):
        self.arm_set = arm_set
        self.reward_vec = reward_vec    
        self.oracle = oracle
        
        self.dim_arms = params["dim_arms"]
        self.num_arms = params["num_arms"]
        self.num_contexts = params["num_contexts"]
        self.num_outcomes = params["num_outcomes"]
        self.reward_vec_norm_ub = params["reward_vec_norm_ub"]
        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.delta= params["failure_level"]
        self.data_path = params["data_path"]

        # self.lmbda = self.num_outcomes*self.dim_arms * np.log(self.horizon /self.delta)/self.reward_vec_norm_ub**2
        self.doubling_constant = 1 #9
        self.lmbda = 0.5
        self.gamma = self.reward_vec_norm_ub * self.param_norm_ub* np.sqrt(self.dim_arms*self.num_outcomes*np.log(self.horizon/self.delta))
        self.e = np.exp(1.0)
        
        self.V_inv = (1.0 / self.lmbda) * np.eye(self.dim_arms*self.num_outcomes)
        self.curr_H = self.lmbda * np.eye(self.dim_arms*self.num_outcomes)
        self.prev_H = 0 * np.eye(self.dim_arms*self.num_outcomes)
        self.t = 1
        self.theta_hat_tau = np.zeros((self.dim_arms*self.num_outcomes,))
        self.non_warm_up_X, self.non_warm_up_Y = [], []
        self.warmup_flag = False
        
        # self.outfile = open(self.data_path + "/outfile.txt" , "w")
        self.regret_arr = []
        self.time_arr = []
        self.batch_times = []
        self.switches_arr = []
        self.hits = 0

    def play_arm(self , arms):
        # Check warm up
        if np.linalg.det(self.curr_H) >= ((1 + self.doubling_constant) * np.linalg.det(self.prev_H)):
            self.time_arr.append(self.batch_times.copy())
            self.batch_times = []
            self.hits += 1
            self.prev_H = self.curr_H.copy()
            
            constraint = [{'type' : "ineq" , "fun": lambda x: self.param_norm_ub**2 - np.dot(x,x)}]
            bounds = [(-self.param_norm_ub , self.param_norm_ub) for _ in range(self.dim_arms*self.num_outcomes)]        
            minimization_res = minimize(log_loss , self.theta_hat_tau, args = (np.array(self.non_warm_up_X), \
                                            np.array(self.non_warm_up_Y), self.lmbda/2, self.dim_arms , self.num_outcomes) , constraints = constraint , bounds = bounds , tol = 0.25*self.num_outcomes)
            thth , succ_flag = minimization_res.x , minimization_res.success
            if succ_flag:
                self.theta_hat_tau = thth
            else:
                print(self.theta_hat_tau)
                print(minimization_res.message)
                assert False
                print("Failed")
                
        self.switches_arr.append(self.hits)
        # Find max UCB arm
        pull_start = time()
        ucb_vals = [self.calculate_UCB(arm) for arm in arms]
        self.batch_times.append(time() - pull_start)
        return arms[np.argmax(ucb_vals)]
    
    def update(self , outcome , played_arm):
        # self.regret_arr.append(regret)
        if outcome != 0:
            self.non_warm_up_X.append(played_arm)
            outcome_one_hot = [0 for _ in range(self.num_outcomes)]
            outcome_one_hot[outcome-1] = 1
            self.non_warm_up_Y.append(outcome_one_hot)

        # Update current H matrix
        # print(np.linalg.det(self.curr_H))
        self.curr_H += np.kron(gradient_MNL(played_arm , self.theta_hat_tau , self.dim_arms , self.num_outcomes) , np.outer(played_arm, played_arm))
        # print("Maximum eigenvalue of gradient: ", np.max(np.linalg.eigvals(gradient_MNL(played_arm , self.theta_hat_tau , self.dim_arms , self.num_outcomes))))
        # print(np.linalg.det(self.curr_H))
        self.t += 1
            

    def play_algorithm(self):
        for t in tqdm(range(self.horizon)):
            arms = self.arm_set[self.t-1] if self.num_contexts == self.horizon else self.arm_set[np.random.choice(self.num_contexts)]
            
            played_arm = self.play_arm(arms)
            best_arm , best_arm_reward = self.find_best_arm_reward(arms)
            self.regret_arr.append(best_arm_reward - self.oracle.expected_reward(played_arm))
            outcome = self.oracle.pull(played_arm)

            update_start = time()
            self.update(outcome , played_arm)
            update_end = time()
            
            # self.outfile.write(f"{self.theta_hat_tau}\n")
            # self.outfile.write(f"DET_current = {np.linalg.det(self.curr_H)} , Regret = {best_arm_reward - self.oracle.expected_reward(played_arm)} , Played_arm = {played_arm} , Best_arm = {best_arm}\n")

            self.batch_times[-1] += update_end - update_start
        
        print(f"Switching Criteria was hit {self.hits} times")
        return self.regret_arr , self.time_arr , self.switches_arr

    def find_best_arm_reward(self , arm_set):
        '''
        finds the best arm with best expected rewards
        '''
        arm_rewards = [self.oracle.expected_reward(arm) for arm in arm_set]
        best_arm_index = np.argmax(arm_rewards)
        return arm_set[best_arm_index] , arm_rewards[best_arm_index]
            

    def calculate_UCB(self , arm):
        # print(prob_vector(arm , self.theta_hat_tau , self.dim_arms , self.num_outcomes).shape)
        return np.dot(self.reward_vec , prob_vector(arm , self.theta_hat_tau , self.dim_arms , self.num_outcomes).reshape(self.num_outcomes,)) + self.error1(arm) + self.error2(arm)
    
    def error1(self , arm):
        return self.gamma * np.sqrt(np.dot(self.reward_vec , np.kron(gradient_MNL(arm , self.theta_hat_tau, self.dim_arms , self.num_outcomes) , arm.T) @ np.linalg.inv(self.curr_H) @ np.kron(gradient_MNL(arm , self.theta_hat_tau, self.dim_arms , self.num_outcomes) , arm) @ self.reward_vec))
    
    def error2(self , arm):
        eigenval , eigenvec = np.linalg.eig(np.kron(np.identity(self.num_outcomes) , np.outer(arm,arm)) @ np.linalg.inv(self.curr_H))
        combination = [(eigenval[i] , eigenvec[:,i]) for i in range(len(eigenval))]
        combination.sort(reverse = True , key = lambda x: x[0])
        max_eigenval = combination[0][0]
        return 3 * self.reward_vec_norm_ub * self.gamma**2 * np.sqrt(max_eigenval)