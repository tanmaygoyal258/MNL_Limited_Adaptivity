import numpy as np
from tqdm import tqdm
from utils import gradient_MNL , prob_vector , log_loss , weighted_norm
from scipy.linalg import sqrtm
from scipy.optimize import minimize

class BatchLinUCB():
    def __init__(self , params , oracle , armset , batch_endpoints , g_distributional_design):
        self.horizon = params["horizon"]
        self.num_batches = int(np.log(np.log(self.horizon))) + 1

        self.num_arms = params["num_arms"]
        self.dim_arms = params["dim_arms"]
        self.num_contexts = params["num_contexts"]
        self.param_norm_ub = params["param_norm_ub"]
        self.failure_level = params["failure_level"]

        self.oracle = oracle
        self.g_distributional_design = g_distributional_design
        self.arms = armset

        self.batch_endpoints = batch_endpoints

        self.theta_estimates = []
        self.lambda_estimates = []

        self.regret_arr = []

        self.alpha = 10 * np.sqrt(np.log(2 * self.dim_arms * self.num_arms * self.horizon / self.failure_level))
        self.outfile = open(params["data_path"] + "/outfile.txt" , "w")

    def play_algorithm(self):
        for batch_num in range(len(self.batch_endpoints)-1):
            print(f"Playing batch {batch_num}")
            theta_update_indices , policy_update_indices = self.divide_indices(batch_num)

            batch_X , batch_rewards , batch_played_arms = self.play_batch(batch_num , theta_update_indices , policy_update_indices)
            
            if self.batch_endpoints[batch_num+1] == self.horizon:
                assert len(self.regret_arr) == self.horizon , f"Length of regret array is {len(self.regret_arr)}"
                return self.regret_arr
            
            print(f"Updating batch parameters")

            self.update_theta_and_Lambda(batch_played_arms , batch_rewards , batch_num)
            
            updated_batch_X = []
            for X in batch_X:
                updated_batch_X.append(self.UCB_LCB_update(X , batch_num))

            self.g_distributional_design.update_parameters(1/self.horizon , updated_batch_X , self.dim_arms)

    
    def play_batch(self , batch_num , theta_update_indices , policy_update_indices):
        batch_indices = [_ for _ in range(self.batch_endpoints[batch_num] , self.batch_endpoints[batch_num+1])]
        batch_X = []
        batch_rewards = []
        batch_played_actions = []

        for t in tqdm(range(len(batch_indices))):
            arms = self.arms[t] if self.num_contexts is None else self.arms[np.random.choice(self.num_contexts)]
            
            updated_arm_set = arms
            for j in range(batch_num):
                updated_arm_set = self.UCB_LCB_update(arms , j)
            
            if t in policy_update_indices:
                batch_X.append(updated_arm_set)    
            
            result = self.g_distributional_design.sample_G_optimal(updated_arm_set) if batch_num == 0 \
                                            else self.g_distributional_design.sample_mixed_softmax(updated_arm_set)
            
            dist_chosen = "g optimal" if batch_num == 0 else result[0]
            dist = result[0] if batch_num == 0 else result[1][0]
            arm_index = result[1] if batch_num == 0 else result[1][1]
            played_arm = result[2] if batch_num == 0 else result[1][2]  

            reward = self.oracle.pull(played_arm)
            
            if t in theta_update_indices:
                batch_rewards.append(reward)
                batch_played_actions.append(played_arm)

            best_arm_index , best_arm , best_arm_reward = self.find_best_arm_reward(arms)
            self.outfile.write(f"best arm index  = {best_arm_index} , Best Arm = {best_arm} , dist_chosen = {dist_chosen} , dist = {dist} , arm_index = {arm_index}, arm_chosen = {played_arm}\n\n")
            self.regret_arr.append(best_arm_reward - self.oracle.expected_reward(played_arm))

        return batch_X , batch_rewards , batch_played_actions

    def UCB_LCB_update(self , arms , index):
        '''
        keeps arms whose UCB estimate is greater than the LCB estimates of all arms
        '''
        theta_estimate = self.theta_estimates[index]
        lambda_estimate = self.lambda_estimates[index]
        # calculate the max LCB
        max_LCB = -np.inf
        for arm in arms:
            max_LCB = max(max_LCB , np.dot(arm , theta_estimate) - self.alpha * weighted_norm(arm , np.linalg.inv(lambda_estimate)))

        # keep the arms whose UCB is greater than the max LCB
        new_arms = []
        for arm in arms:
            if (np.dot(arm , theta_estimate) + self.alpha * weighted_norm(arm , np.linalg.inv(lambda_estimate))) > max_LCB:
                new_arms.append(arm)
        
        return new_arms

    def divide_indices(self , batch_num):
        '''
        returns the indices for updating the theta and policy
        '''
        batch_indices = [_ - self.batch_endpoints[batch_num] for _ in range(self.batch_endpoints[batch_num] , self.batch_endpoints[batch_num+1])]
        theta_update_indices = []
        policy_update_indices = []
        # theta_update_indices = np.random.choice(batch_indices , int(len(batch_indices)/2) , replace = False).tolist()
        # policy_update_indices = [_ for _ in batch_indices if _ not in theta_update_indices]
        for i in batch_indices:
            choice = np.random.choice([0,1])
            if choice == 0:
                theta_update_indices.append(i)
            else:
                policy_update_indices.append(i)
        return theta_update_indices , policy_update_indices
    
    def update_theta_and_Lambda(self , played_arms , rewards , batch_num):
        
        lamda = 32 * np.log(2 * self.dim_arms * self.horizon / self.failure_level)
        lambda_updated = lamda * np.identity(self.dim_arms)
        for arm in played_arms:
            lambda_updated += np.outer(arm , arm)

        zhi = rewards[0] * played_arms[0]
        for reward , arm in zip(rewards[1:] , played_arms[1:]):
            zhi += reward * arm
        
        theta_updated = np.linalg.inv(lambda_updated) @ zhi

        self.theta_estimates.append(theta_updated)
        self.lambda_estimates.append(lambda_updated)

    def find_best_arm_reward(self , arm_set):
        '''
        finds the best arm with best expected rewards
        '''
        arm_rewards = [self.oracle.expected_reward(arm) for arm in arm_set]
        best_arm_index = np.argmax(arm_rewards)
        return best_arm_index , arm_set[best_arm_index] , arm_rewards[best_arm_index]
