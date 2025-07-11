import numpy as np
from optimization import fit_online_logistic_estimate, fit_online_logistic_estimate_bar
from utils import *
from tqdm import tqdm

class ada_OFU_ECOLog():
    def __init__(self, params , oracle):
        
        # initializing the reward functions and oracles
        self.reward_type = "logistic"
        self.reward_func = sigmoid if self.reward_type == "logistic" else probit
        self.d_reward_func = dsigmoid if self.reward_type == "logistic" else dprobit
        self.oracle = oracle

        # setting the variables from params
        self.num_contexts = params["num_contexts"]
        self.item_count = params["num_arms"]
        self.horizon = params["horizon"]
        self.dim = params["dim_arms"]
        self.l2reg = 5
        self.failure_level = params["failure_level"]
        self.param_norm_ub = params["param_norm_ub"]
        self.number_arms = params["num_arms"]

        # initializing the arm set
        if self.num_contexts != self.horizon:
            self.arm_set = self.create_arm_set(np.random.default_rng(params["arm_seed"]))
        self.arm_rng = np.random.default_rng(params["arm_seed"])

        # initialzing the matrices and thetas
        self.vtilde_matrix = self.l2reg * np.eye(self.dim)
        self.vtilde_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.theta = np.zeros((self.dim,))
        self.conf_radius = 0
        self.cum_loss = 0
        self.ctr = 1

        self.regret_arr = []
        self.time_arr = []

        np.random.seed(0)

    def update_parameters(self, arm, reward):
        # compute new estimate theta
        self.theta = np.real_if_close(fit_online_logistic_estimate(arm=arm,
                                                                reward=reward,
                                                                current_estimate=self.theta,
                                                                vtilde_matrix=self.vtilde_matrix,
                                                                vtilde_inv_matrix=self.vtilde_matrix_inv,
                                                                constraint_set_radius=self.param_norm_ub,
                                                                diameter=self.param_norm_ub,
                                                                precision=1/self.ctr))
        # compute theta_bar (needed for data-dependent conf. width)
        theta_bar = np.real_if_close(fit_online_logistic_estimate_bar(arm=arm,
                                                                    current_estimate=self.theta,
                                                                    vtilde_matrix=self.vtilde_matrix,
                                                                    vtilde_inv_matrix=self.vtilde_matrix_inv,
                                                                    constraint_set_radius=self.param_norm_ub,
                                                                    diameter=self.param_norm_ub,
                                                                    precision=1/self.ctr))
        disc_norm = np.clip(mat_norm(self.theta-theta_bar, self.vtilde_matrix), 0, np.inf)

        # update matrices
        sensitivity = self.d_reward_func(np.dot(self.theta, arm))
        self.vtilde_matrix += sensitivity * np.outer(arm, arm)
        self.vtilde_matrix_inv += - sensitivity * np.dot(self.vtilde_matrix_inv,
                                                        np.dot(np.outer(arm, arm), self.vtilde_matrix_inv)) / (
                                          1 + sensitivity * np.dot(arm, np.dot(self.vtilde_matrix_inv, arm)))
        
        # sensitivity check
        sensitivity_bar = self.d_reward_func(np.dot(theta_bar, arm))
        if sensitivity_bar / sensitivity > 2:
            msg = f"\033[95m Oops. ECOLog has a problem: the data-dependent condition was not met. This is rare; try increasing the regularization (self.l2reg) \033[95m"
            raise ValueError(msg)

        # update sum of losses and ctr
        coeff_theta = self.reward_func(np.dot(self.theta, arm))
        loss_theta = -reward * np.log(coeff_theta) - (1-reward) * np.log(1-coeff_theta)
        coeff_bar = self.reward_func(np.dot(theta_bar, arm))
        loss_theta_bar = -reward * np.log(coeff_bar) - (1-reward) * np.log(1-coeff_bar)
        self.cum_loss += 2*(1+self.param_norm_ub)*(loss_theta_bar - loss_theta) - 0.5*disc_norm
        self.ctr += 1

    def pull(self, arm_set):
        # bonus-based version (strictly equivalent to param-based for this algo) of OL2M
        self.update_ucb_bonus()
        # compute optimistic rewards
        picked_embedding_index = self.find_argmax(arm_set)   
        return picked_embedding_index

    def update_ucb_bonus(self):
        """
        Updates the ucb bonus function (a more precise version of Thm3 in ECOLog paper, refined for the no-warm up alg)
        """
        gamma = np.sqrt(self.l2reg) / 2 + 2 * np.log(
            2 * np.sqrt(1 + self.ctr / (4 * self.l2reg)) / self.failure_level) / np.sqrt(self.l2reg)
        res_square = 2*self.l2reg*self.param_norm_ub**2 + (1+self.param_norm_ub)**2*gamma + self.cum_loss
        res_square = max(0 , res_square)
        self.conf_radius = np.sqrt(res_square)

    def compute_optimistic_reward(self, arm):
        """
        Returns prediction + exploration_bonus for arm.
        """
        norm = mat_norm(arm, self.vtilde_matrix_inv)
        pred_reward = self.reward_func(np.sum(self.theta * arm))
        bonus = self.conf_radius * norm
        return pred_reward + bonus

    def find_argmax(self, arm_set):
        """
        Returns the arm that maximizes the optimistic reward.
        """
        arm_values = [self.compute_optimistic_reward(arm.reshape(-1,)) for arm in arm_set]
        return [np.argmax(arm_values)]
    
    def play_algorithm(self):
        for t in tqdm(range(self.horizon)):
            
            # obtain the arms
            if self.num_contexts != self.horizon:
                arm_set = self.slot_arms[np.random.choice(self.num_contexts)]
            else:
                arm_set = self.create_arm_set(self.arm_rng)

            picked_arm = arm_set[self.pull(arm_set)[0]].reshape(-1,)

            # obtain the actual reward and expected regret
            best_arm , best_arm_reward = self.find_best_arm_reward(arm_set)
            actual_reward = self.oracle.pull(picked_arm)    # either 0 or 1
            expected_regret = best_arm_reward - self.oracle.expected_reward(picked_arm)

            # update the parameters
            self.update_parameters(picked_arm , actual_reward) 

            # store the regrets, rewards, and time
            self.regret_arr.append(expected_regret)
            self.ctr += 1

        return self.regret_arr
    

    def find_best_arm_reward(self , arm_set):
        '''
        finds the best arm with best expected rewards
        '''
        arm_rewards = [self.oracle.expected_reward(arm) for arm in arm_set]
        best_arm_index = np.argmax(arm_rewards)
        return arm_set[best_arm_index] , arm_rewards[best_arm_index]

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