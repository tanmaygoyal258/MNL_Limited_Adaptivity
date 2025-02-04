import numpy as np
from tqdm import tqdm
from utils import gradient_MNL , prob_vector , weighted_norm , minimize_theta_function
from scipy.linalg import sqrtm
from scipy.optimize import minimize

class MLogB():

    def __init__(self , params , oracle , armset):
        self.horizon = params["horizon"]
        self.num_batches = params["num_batches"]
        self.dim_arms = params["dim_arms"]
        self.param_norm_ub = params["param_norm_ub"]
        self.reward_vec_norm_ub = params["reward_vec_norm_ub"]
        self.num_outcomes = params["num_outcomes"]
        self.reward_vec = params["reward_vec"]
        self.failure_level = params["failure_level"]
        self.num_contexts = params["num_contexts"]

        self.oracle = oracle
        self.arms = armset

        self.regret_arr = []

        self.alpha = 2*(1+self.param_norm_ub) + np.log(self.num_outcomes + 1)
        self.lamda = np.sqrt(self.num_outcomes) * self.alpha * self.param_norm_ub
        self.eta = self.param_norm_ub/2 + np.log(self.num_outcomes+1)/4

        self.theta = [np.random.random()*2-1 for _ in range(self.dim_arms*self.num_outcomes)]
        self.theta = np.array(self.theta)/np.linalg.norm(self.theta) * self.param_norm_ub
        self.H = self.lamda * np.identity(self.dim_arms*self.num_outcomes)
        

    def play_algorithm(self):

        for t in tqdm(range(self.horizon)):
            
            # obtain the arms for the particular round
            arms = self.arms[t] if self.num_contexts is None else self.arms[np.random.choice(self.num_contexts)]
            
            # pull the arm
            played_arm = self.pull(arms , t)
            # obtain the reward and regret
            outcome , reward = self.oracle.pull(played_arm)
            best_arm , best_arm_reward = self.find_best_arm_reward(arms)
            self.regret_arr.append(best_arm_reward - self.oracle.expected_reward(played_arm))

            # update H tilde
            H_tilde = self.H.copy() + self.eta * np.kron(gradient_MNL(played_arm , self.theta) , np.outer(played_arm,played_arm))
            # update theta
            self.update_theta(H_tilde , played_arm , outcome)
            # update H_t
            self.H += np.kron(gradient_MNL(played_arm , self.theta) , np.outer(played_arm,played_arm))

        return self.regret_arr
    
    def find_best_arm_reward(self , arm_set):
        '''
        finds the best arm with best expected rewards
        '''
        arm_rewards = [self.oracle.expected_reward(arm) for arm in arm_set]
        best_arm_index = np.argmax(arm_rewards)
        return arm_set[best_arm_index] , arm_rewards[best_arm_index]
    
    def pull(self , arm_set , t):
        '''
        pulls the arms based on a UCB estimate
        '''
        self.beta = np.log(self.num_outcomes) * np.log(t+1) * np.sqrt(self.num_outcomes*self.dim_arms)
        arm_values = [self.compute_optimistic_reward(arm) for arm in arm_set]
        return arm_set[np.argmax(arm_values)]
    
    def compute_optimistic_reward(self , arm):
        '''
        computes the optimstic reward
        '''
        arm = np.array(arm.copy()).reshape((self.dim_arms , 1))
        estimated_reward = np.dot(self.reward_vec , prob_vector(arm , self.theta))

        error_term_1 = self.beta * \
                        np.linalg.norm(sqrtm(np.linalg.inv(self.H)) @ \
                                    np.kron(np.identity(self.num_outcomes) , arm) @ \
                                    gradient_MNL(arm , self.theta) @ self.reward_vec)
        
        error_term_2 = 3 * self.reward_vec_norm_ub * self.beta**2 * \
                        np.linalg.norm(np.kron(np.identity(self.num_outcomes) , arm.T) @ \
                                    sqrtm(np.linalg.inv(self.H)) , ord = 2)**2
        
        return estimated_reward + error_term_1 + error_term_2
    
    def update_theta(self , H_tilde , arm , outcome):
        '''
        updates theta using online mirror descent
        '''
        outcome_vector = [0 for _ in range(self.num_outcomes)]
        if outcome != 0:
            outcome_vector[outcome-1] = 1
        outcome_vector = np.array(outcome_vector)
        loss_gradient = np.kron(outcome_vector - prob_vector(arm , self.theta) , arm) - self.lamda * self.theta
        # find Z
        Z = self.theta - self.eta * np.linalg.inv(H_tilde) @ loss_gradient

        # new theta is the eigenvector corresponding to min eigenvalue
        eigvals , eigvecs = np.linalg.eig(np.linalg.inv(H_tilde))

        combination = [(eigvals[i] , eigvecs[:,i]) for i in range(len(eigvals))]
        combination.sort(reverse = False , key = lambda x: x[0])
        min_eigvec = combination[0][1]

        MD_theta = min_eigvec + Z
        MD_theta /= np.linalg.norm(MD_theta)
        MD_theta *= self.param_norm_ub        

        constraint = [{'type' : "ineq" , "fun": lambda x: self.param_norm_ub-np.linalg.norm(x)}]
        scipy_theta = minimize(minimize_theta_function , self.theta , args =(Z , H_tilde) , constraints = constraint).x
        self.theta = scipy_theta.copy()

        # print(f"MD {MD_theta} , scipy {scipy_theta}")