"""
Created on 10/22/23
@author: nicklee

Class for the OFULog+ of [Lee et al., AISTATS'24]. Inherits from the LogisticBandit class.

Additional Attributes
---------------------
lazy_update_fr : int
    integer dictating the frequency at which to do the learning if we want the algo to be lazy
theta_hat : np.array(dim)
    maximum-likelihood estimator
log_loss_hat : float
    log-loss at current estimate theta_hat
ctr : int
    counter for lazy updates
"""
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from utils import sigmoid
# from logbexp.algorithms.logistic_bandit_algo import LogisticBandit


def mu(z):
    return 1 / (1 + np.exp(-z))


class OFULogPlus():
    def __init__(self, params , oracle, lazy_update_fr=1, tol=1e-7):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.ucb_bonus = 0
        self.log_loss_hat = 0
        self.tol = tol
        

        self.oracle = oracle
        self.item_count = params["num_arms"]
        self.horizon = params["horizon"]
        self.dim = params["dim_arms"]
        self.num_contexts = params["num_contexts"]
        self.failure_level = params["failure_level"]
        self.param_norm_ub = params["param_norm_ub"]
        self.number_arms = params["num_arms"]

        # initializing the arm set
        if self.num_contexts != self.horizon:
            self.arm_set = self.create_arm_set(np.random.default_rng(params["arm_seed"]))
        self.arm_rng = np.random.default_rng(params["arm_seed"])

        self.theta_hat = np.zeros((self.dim,))
        self.ctr = 1

        self.regret_arr = []
        self.rewards = np.zeros((0,))
        self.arms = np.zeros((0, self.dim))

    def update_parameters(self, arm, reward):
        """
        Updates estimator.
        """
        self.arms = np.vstack((self.arms, arm))
        self.rewards = np.concatenate((self.rewards, [reward]))

        ## SLSQP
        ineq_cons = {'type': 'ineq',
                    'fun': lambda theta: np.array([
                         self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                     'jac': lambda theta: 2 * np.array([- theta])}
        opt = minimize(self.neg_log_likelihood_full, x0=np.reshape(self.theta_hat, (-1,)), method='SLSQP',
                    jac=self.neg_log_likelihood_full_J,
                    constraints=ineq_cons, tol=self.tol)
        self.theta_hat = opt.x

        # update counter
        self.ctr += 1

    def pull(self, arm_set):
        self.update_ucb_bonus()
        self.log_loss_hat = self.neg_log_likelihood_full(self.theta_hat)
        picked_embedding_index = self.find_argmax(arm_set)   
        return picked_embedding_index


    def find_argmax(self, arm_set):
        """
        Returns the arm that maximizes the optimistic reward.
        """
        arm_values = [self.compute_optimistic_reward(arm.reshape(-1,)) for arm in arm_set]
        return [np.argmax(arm_values)]

    def update_ucb_bonus(self):
        self.ucb_bonus = 10 * self.dim * np.log(
            np.e + (self.param_norm_ub * len(self.rewards) / (2 * self.dim))) + 2 * (
                                 np.e - 2 + self.param_norm_ub) * np.log(1 / self.failure_level)

    def compute_optimistic_reward(self, arm):
        if self.ctr == 1:
            res = np.linalg.norm(arm)
        else:
            ## SLSQP
            obj = lambda theta: -np.sum(arm * theta)
            obj_J = lambda theta: -arm
            ineq_cons = {'type': 'ineq',
                        'fun': lambda theta: np.array([
                            self.ucb_bonus - (self.neg_log_likelihood_full(theta) - self.log_loss_hat),
                             self.param_norm_ub ** 2 - np.dot(theta, theta)]),
                         'jac': lambda theta: - np.vstack((self.neg_log_likelihood_full_J(theta).T, 2 * theta))}
            opt = minimize(obj, x0=self.theta_hat, method='SLSQP', jac=obj_J, constraints=ineq_cons, tol=self.tol)
            res = np.sum(arm * opt.x)
        return res
    
    def play_algorithm(self):
        for t in tqdm(range(self.horizon)):
            
            # obtain the arms
            if self.num_contexts != self.horizon:
                arm_set = self.slot_arms[np.random.choice(self.num_contexts)]
            else:
                arm_set = self.create_arm_set(self.arm_rng)

            # pull the arm
            picked_arm = arm_set[self.pull(arm_set)[0]].reshape(-1,)

            # obtain the actual reward and expected regret
            best_arm , best_arm_reward = self.find_best_arm_reward(arm_set)
            actual_reward = self.oracle.pull(picked_arm)
            expected_regret = best_arm_reward - self.oracle.expected_reward(picked_arm)

            # update the parameters
            self.update_parameters(picked_arm , actual_reward) 

            # store the regrets, and rewards
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
    
    def neg_log_likelihood_full(self, theta):
        """
        Computes the full log-loss at theta
        """
        return self.neg_log_likelihood(theta, self.arms, self.rewards)
    
    def neg_log_likelihood(self, theta, arms, rewards):
        """
        """
        if len(rewards) == 0:
            return 0
        else:
            arms_theta = arms @ theta
            return - np.sum(rewards * np.log(sigmoid(arms_theta)) + (1 - rewards) * np.log(sigmoid(- arms_theta)))

    def neg_log_likelihood_full_J(self, theta):
        """
        Derivative of neg_log_likelihood_full
        """
        return self.neg_log_likelihood_J(theta, self.arms, self.rewards)
    
    def neg_log_likelihood_J(self, theta, arms, rewards):
        """
        Derivative of neg_log_likelihood
        """
        if len(rewards) == 0:
            return np.zeros(self.dim)
        else:
            return arms.T @ (sigmoid(arms @ theta) - rewards)

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