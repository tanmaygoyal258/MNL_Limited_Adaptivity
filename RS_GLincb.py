import numpy as np
from utils import mat_norm, solve_glm_mle, dsigmoid, dprobit
from tqdm import tqdm
from time import time

class RS_GLinUCB:
    def __init__(self, params , arm_set, kappa , oracle):
        self.arm_set = arm_set
        self.oracle = oracle
        
        self.dim_arms = params["dim_arms"]
        self.num_arms = params["num_arms"]
        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.delta= params["failure_level"]
        self.kappa = kappa   
        self.data_path = params["data_path"]
        self.num_contexts = params["num_contexts"]
        
        self.model = "Logistic"    
        self.logistic_constant = 1 # TODO
        
        self.doubling_constant = 0.5
        self.lmbda = 0.5 #self.dim_arms * np.log(self.horizon /self.delta)/self.logistic_constant**2
        self.gamma = self.logistic_constant * self.param_norm_ub* np.sqrt(self.dim_arms*np.log(self.horizon/self.delta))
        self.beta = 4* np.sqrt(self.dim_arms* np.log(self.horizon/self.delta))
        self.warmup_threshold = 1.0 / (0.01 * self.kappa * self.logistic_constant**4 * self.param_norm_ub**2 * self.dim_arms)
        
        self.V_inv = (1.0 / self.lmbda) * np.eye(self.dim_arms)
        self.curr_H = self.lmbda * np.eye(self.dim_arms)
        self.prev_H = 0 * np.eye(self.dim_arms)
        self.theta_hat_w = np.zeros((self.dim_arms,))
        self.theta_hat_tau = np.zeros((self.dim_arms,))
        self.warm_up_X, self.warm_up_Y = [], []
        self.non_warm_up_X, self.non_warm_up_Y = [], []
        self.warmup_flag = False
        
        self.t = 1
        self.e = np.exp(1.0)
        
        self.regret_arr = []
        self.time_arr = []
        self.batch_times = []
        # self.outfile = open(self.data_path + "/outfile.txt" , "w")

    def play_arm(self , arms):
        # Check warm up
        self.warmup_flag = False
        max_x, max_norm, max_ind = None, 0, -1
        for i in range(self.num_arms):
            x = arms[i]
            mnorm = mat_norm(x, self.V_inv)**2
            if mnorm > self.warmup_threshold:
                if mnorm > max_norm:
                    max_x = x
                    max_norm = mnorm
                    max_ind = i
                self.warmup_flag = True
        if self.warmup_flag:
            max_x_V_inv = np.dot(self.V_inv, max_x)
            self.V_inv -= np.outer(max_x_V_inv, max_x_V_inv) / (1.0 + max_norm)
            self.a_t = max_ind
        
        # Non warm-up
        else:
            # Check determinant
            if np.linalg.det(self.curr_H) >= ((1 + self.doubling_constant) * np.linalg.det(self.prev_H)):
                self.time_arr.append(self.batch_times.copy())
                self.batch_times = []
                self.prev_H = self.curr_H
                thth, succ_flag = solve_glm_mle(self.theta_hat_tau, np.array(self.non_warm_up_X), \
                                                np.array(self.non_warm_up_Y), self.lmbda/2, self.model)
                if succ_flag:
                    self.theta_hat_tau = thth
                else:
                    print("Failed")
            
            pull_start = time()
            # Eliminate arms based on warm-up theta
            max_lcb = -np.inf
            arm_idx = []
            ucb_arr = []
            # Compute LCB and UCB of each arm
            for i in range(self.num_arms):
                lcb_idx = np.dot(self.theta_hat_w, arms[i]) \
                        - self.gamma * np.sqrt(self.kappa) * mat_norm(arms[i], self.V_inv)
                ucb_idx = np.dot(self.theta_hat_w, arms[i]) \
                        + self.gamma * np.sqrt(self.kappa) * mat_norm(arms[i], self.V_inv)
                ucb_arr.append(ucb_idx)
                if lcb_idx > max_lcb:
                    max_lcb = lcb_idx
            # Eliminate arms
            for i in range(self.num_arms):
                if ucb_arr[i] >= max_lcb:
                    arm_idx.append(i)
            # Find max UCB arm
            max_ind = -np.inf
            self.a_t = -1
            for i in arm_idx:
                ucb_ind = np.dot((self.theta_hat_w + self.theta_hat_tau)/2, arms[i]) \
                        + self.beta * mat_norm(arms[i], np.linalg.inv(self.prev_H))
                if ucb_ind > max_ind:
                    max_ind = ucb_ind
                    self.a_t = i
            self.batch_times.append(time() - pull_start)
        return self.a_t
    
    def update(self , reward , arms):
        # self.regret_arr.append(regret)
        # If this was a warm up round, append (x, y) to warm-up set (and non-warm-up set), re-compute theta_hat_w
        if self.warmup_flag:
            self.warm_up_X.append(arms[self.a_t])
            self.warm_up_Y.append(reward)
            self.non_warm_up_X.append(arms[self.a_t])
            self.non_warm_up_Y.append(reward)
            thth, succ_flag = solve_glm_mle((self.theta_hat_w + self.theta_hat_tau)/2, np.array(self.warm_up_X), \
                                                np.array(self.warm_up_Y), self.lmbda/2, self.model)
            if succ_flag:
                self.theta_hat_w = thth # update theta_hat_w if mle solution was successful
        
        # Else add (x, y) to non warm-up set
        else:
            self.non_warm_up_X.append(arms[self.a_t])
            self.non_warm_up_Y.append(reward)
            if self.model == 'Logistic':
                mudp = dsigmoid(np.dot(arms[self.a_t], self.theta_hat_w))
            elif self.model == 'Probit':
                mudp = dprobit(np.dot(arms[self.a_t], self.theta_hat_w))
            # Update current H matrix
            self.curr_H += mudp  * np.outer(arms[self.a_t], arms[self.a_t])
        
        self.t += 1
            

    def play_algorithm(self):
        for t in tqdm(range(self.horizon)):
            arms = self.arm_set[self.t-1] if self.num_contexts == self.horizon else self.arm_set[np.random.choice(self.num_contexts)]
            arms = [arm.reshape(-1,) for arm in arms]
            played_arm = arms[self.play_arm(arms)]
            best_arm , best_arm_reward = self.find_best_arm_reward(arms)
            self.regret_arr.append(best_arm_reward - self.oracle.expected_reward(played_arm))
            actual_reward = self.oracle.pull(played_arm)
            update_start = time()
            self.update(actual_reward , arms)
            update_end = time()
            self.batch_times[-1] += update_end - update_start
            # self.outfile.write(f"{self.theta_hat_tau} , {self.theta_hat_w} \n")
            # self.outfile.write(f"Reward = {actual_reward} , Regret = {best_arm_reward - self.oracle.expected_reward(played_arm)} , Played_arm = {played_arm} , Best_arm = {best_arm}\n")
        return self.regret_arr , self.time_arr

    def find_best_arm_reward(self , arm_set):
        '''
        finds the best arm with best expected rewards
        '''
        arm_rewards = [self.oracle.expected_reward(arm) for arm in arm_set]
        best_arm_index = np.argmax(arm_rewards)
        return arm_set[best_arm_index] , arm_rewards[best_arm_index]
            

