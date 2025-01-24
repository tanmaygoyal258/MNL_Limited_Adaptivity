import numpy as np
from tqdm import tqdm
from utils import gradient_MNL , prob_vector , log_loss
from scipy.linalg import sqrtm
from scipy.optimize import minimize

class B_MNL():
    def __init__(self , params , oracle , armset , batch_endpoints , g_distributional_design):
        self.horizon = params["horizon"]
        self.num_batches = params["num_batches"]
        self.dim_arms = params["dim_arms"]
        self.param_norm_ub = params["param_norm_ub"]
        self.reward_vec_norm_ub = params["reward_vec_norm_ub"]
        self.num_outcomes = params["num_outcomes"]
        self.reward_vec = params["reward_vec"]
        self.failure_level = params["failure_level"]

        self.oracle = oracle
        self.g_distributional_design = g_distributional_design
        self.arms = armset

        self.batch_endpoints = batch_endpoints

        self.theta_estimates = []
        self.H_estimates = []
        # self.V_estimates = []

        self.regret_arr = []

        self.lamda = 0.5
        self.confidence_radius = self.param_norm_ub * self.reward_vec_norm_ub * np.sqrt(self.dim_arms*np.log(self.horizon/self.failure_level))


    def play_algorithm(self):
        for batch_num in range(len(self.batch_endpoints)-1):
            print(f"Playing batch {batch_num}")
            batch_X , batch_rewards , batch_played_arms , batch_outcomes = self.play_batch(batch_num)
            
            if self.batch_endpoints[batch_num+1] == self.horizon:
                assert len(self.regret_arr) == self.horizon , f"Length of regret array is {len(self.regret_arr)}"
                return self.regret_arr
            
            theta_update_indices , policy_update_indices = self.divide_indices(batch_num)

            required_rewards = [batch_rewards[i] for i in theta_update_indices]
            required_arms = [batch_played_arms[i] for i in theta_update_indices]
            required_outcomes = [batch_outcomes[i] for i in theta_update_indices]
            required_X = [batch_X[i] for i in policy_update_indices]

            print(f"Updating batch parameters")

            self.update_theta_and_H(required_arms , required_rewards , required_outcomes , batch_num)
            self.g_distributional_design.update_parameters(1/self.horizon , self.num_outcomes , required_X , self.dim_arms , self.theta_estimates[-1] , self.self_concordance_factor)

    
    def play_batch(self , batch_num):
        batch_indices = [_ for _ in range(self.batch_endpoints[batch_num] , self.batch_endpoints[batch_num+1])]
        batch_X = []
        batch_rewards = []
        batch_played_actions = []
        batch_outcomes = []

        for t in tqdm(batch_indices):
            arms = self.arms[t]
            
            updated_arm_set = arms
            for j in range(batch_num):
                updated_arm_set = self.UCB_LCB_update(arms , j)
            batch_X.append(updated_arm_set)    
            played_arm = self.g_distributional_design.sample_G_optimal(updated_arm_set) if batch_num == 0 \
                        else self.g_distributional_design.sample_MNL_policy(updated_arm_set , self.num_outcomes , self.theta_estimates[-1] , self.self_concordance_factor)
            outcome , reward = self.oracle.pull(played_arm)

            batch_outcomes.append(outcome)
            batch_rewards.append(reward)
            batch_played_actions.append(played_arm)

            best_arm , best_arm_reward = self.find_best_arm_reward(arms)
            self.regret_arr.append(best_arm_reward - self.oracle.expected_reward(played_arm))

        return batch_X , batch_rewards , batch_played_actions , batch_outcomes

    def UCB_LCB_update(self , arms , index):
        '''
        keeps arms whose UCB estimate is greater than the LCB estimates of all arms
        '''
        theta_estimate = self.theta_estimates[index]
        H_estimate = self.H_estimates[index]
        # calculate the max LCB
        max_LCB = -np.inf
        for arm in arms:
            max_LCB = max(max_LCB , np.dot(self.reward_vec , prob_vector(arm , theta_estimate)) - self.error_term1(theta_estimate , arm , H_estimate) - self.error_term2(theta_estimate , arm , H_estimate))
        # keep the arms whose UCB is greater than the max LCB
        new_arms = []
        for arm in arms:
            if np.dot(self.reward_vec , prob_vector(arm , theta_estimate)) + self.error_term1(theta_estimate , arm , H_estimate) + self.error_term2(theta_estimate , arm , H_estimate) > max_LCB:
                new_arms.append(arm)
        
        return new_arms

    def error_term1(self , theta , arm , H):
        return self.reward_vec_norm_ub * self.confidence_radius * np.linalg.norm(sqrtm(gradient_MNL(arm , theta)) @ np.kron(np.identity(self.num_outcomes) , arm.T) @ sqrtm(np.linalg.inv(H)) , ord = 2)
    
    def error_term2(self , theta , arm , H):
        return 3 * self.param_norm_ub * self.confidence_radius**2 * np.linalg.norm(np.kron(np.identity(self.num_outcomes) , arm.T) @ sqrtm(np.linalg.inv(H)) , ord = 2)**2
    
    def divide_indices(self , batch_num):
        '''
        returns the indices for updating the theta and policy
        '''
        batch_indices = [_ - self.batch_endpoints[batch_num] for _ in range(self.batch_endpoints[batch_num] , self.batch_endpoints[batch_num+1])]
        theta_update_indices = np.random.choice(batch_indices , int(len(batch_indices)/2) , replace = False).tolist()
        policy_update_indices = [_ for _ in batch_indices if _ not in theta_update_indices]
        return theta_update_indices , policy_update_indices
    
    def update_theta_and_H(self , played_arms , rewards , outcomes , batch_num):
        theta_previous = np.zeros((self.dim_arms * self.num_outcomes)) if batch_num == 0 else self.theta_estimates[batch_num-1]
        theta_updated = minimize(log_loss , theta_previous , args = (played_arms , outcomes , self.lamda)).x

        H_updated = self.lamda * np.identity(self.num_outcomes * self.dim_arms)
        for arm in played_arms:
            H_updated += np.kron(gradient_MNL(arm , theta_updated) , np.outer(arm , arm)) / self.self_concordance_factor(arm , theta_updated)
        
        self.theta_estimates.append(theta_updated)
        self.H_estimates.append(H_updated)

    def self_concordance_factor(self , arm , theta):
        return np.exp(np.sqrt(6) * 2 * self.param_norm_ub)
    
    def find_best_arm_reward(self , arm_set):
        '''
        finds the best arm with best exoected rewards
        '''
        arm_rewards = [self.oracle.expected_reward(arm) for arm in arm_set]
        best_arm_index = np.argmax(arm_rewards)
        return arm_set[best_arm_index] , arm_rewards[best_arm_index]
