import numpy as np
from scipy.optimize import minimize , LinearConstraint
from scipy.stats import norm

def information_matrix_set(distribution_X , X):
    '''
    returns the information matrix for the given distribution over X
    '''
    assert len(distribution_X) == len(X)
    soln = np.zeros((len(X[0]) , len(X[0])))
    for p,x in zip(distribution_X , X):
        soln += p * np.outer(x , x)
    return soln

def information_matrix_distibution(distribution_D  , D):
    '''
    returns the information matrix for a given distribution over D which is a distribution over X
    '''
    assert distribution_D.shape[0] == D.shape[0]
    soln = information_matrix_set(distribution_D[0] , D[0])
    for d_X , X in enumerate(zip(D[1:] , distribution_D[1:])):
        soln += information_matrix_set(d_X , X)
    return soln

def weighted_norm(x , A):
    return np.sqrt(np.dot(x , np.dot(A , x)))

def g_optimal_design(X , algorithm = "d_optimal" , BS = None):
    '''
    returns the g optimal design for a given set of points
    '''
    # try:
    #     dim = len(X[0])
    # except:
    #     print(X)
    #     return None
    # if algorithm == "barycentric_spanner":
    #     assert BS is not None
    #     spanning_set =  BarycentricSpanner(dim , X , BS).spanning_set
    #     assert spanning_set.shape[0] == dim
    #     distribution = [0 for _ in range(len(X))]
    #     for i in range(len(X)):
    #         if X[i] in spanning_set:
    #             distribution[i] = 1
    #     return [i/dim for i in distribution]
    
    # elif algorithm == "d_optimal":
    #     from D_optimal import D_Optimal
    #     spanning_set = D_Optimal(dim , X)
    #     distribution = spanning_set.d_optimal_policy
    #     return distribution
    
    # else:
    #     assert False
    num_arms = len(X)
    p_0 = [np.random.random() for i in range(num_arms)]
    p_0 = np.array(p_0) / np.sum(p_0)
    constraint = LinearConstraint(np.ones(num_arms) , lb = 1 , ub = 1)
    bound = [(0,1) for _ in range(num_arms)]
    g_opt_design = minimize(d_optimal_objective , p_0 , args = (X,) , constraints = constraint , bounds = bound ).x
    g_opt_design /= np.sum(g_opt_design)
    return g_opt_design

def sample_softmax(X , M , alpha):
    '''
    samples an arm from X with a alpha-softmax distribution on the weighted norms with M<
    '''
    weights = [weighted_norm(x , M)**2 for x in X]
    prob_dist_unnormalized = np.array(weights) ** alpha
    prob_dist_normalized = prob_dist_unnormalized / np.sum(prob_dist_unnormalized)
    arm_index = np.random.choice(len(X) , p = prob_dist_normalized)
    return prob_dist_normalized , arm_index , X[arm_index]

def prob_vector(arm , theta , dim , num_outcomes):
    '''
    returns a softmax probability vector with an additional 1 (=exp(0)) in the denominator to account for no action chosen
    '''
    theta_temp = np.reshape(theta , (dim , num_outcomes))
    if len(arm.shape) == 1: # no arms in the matrix
        return None
    if len(arm.shape) == 3: # multiple arms together
        arm =  arm.reshape((arm.shape[0] , arm.shape[1]))
    else: # only one arm at a time
        arm = arm.reshape(1,dim)
    inner_products = arm @ theta_temp
    num_entries = inner_products.shape[0]
    # adding the 0 for no action chosen
    inner_products = np.hstack([np.array([0 for _ in range(num_entries)]).reshape(num_entries , 1) , inner_products])
    exponent = np.exp(inner_products)
    sum_exponents_inv = 1/np.sum(exponent , axis = 1).reshape(-1,1)
    return (sum_exponents_inv * exponent)[: , 1:]

def gradient_MNL(arm , theta , dim , num_outcomes):
    '''
    returns the gradient of the probability vector with resepect to an arm and theta
    ''' 
    z = prob_vector(arm , theta , dim , num_outcomes)[0]
    return np.diag(z) - np.outer(z , z)


def log_loss(theta , arms , outcomes , lamda , dim , num_outcomes):
    loss = 0
    probability_vectors = prob_vector(arms , theta  , dim , num_outcomes)
    outcomes = np.array(outcomes.T)
    loss = -np.sum(np.log(probability_vectors@outcomes+1e-12)) if probability_vectors is not None else 0
    # for arm , outcome in zip(arms , outcomes):    #     loss -= np.log(prob_vector(arm , theta , dim , num_outcomes)[outcome-1])
    return loss + lamda/2 * np.linalg.norm(theta)**2


def d_optimal_objective(distribution_X , X):
    '''
    returns the log of the determinant of information matrix for the given distribution over X
    '''
    return -np.log(np.linalg.det(information_matrix_set(distribution_X , X)) + 1e-12)


def minimize_omd_loss(theta , theta_prev , loss_gradient , eta , H_tilde):
    return np.dot(loss_gradient , theta) +  1/(2*eta) * weighted_norm(theta - theta_prev , H_tilde)**2

#############################################

def mat_norm(vec, matrix):
    return np.sqrt(np.dot(vec, np.dot(matrix, vec)))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def log_loss_glm(theta, X, Y, lmbda, model):
    if model == 'Logistic':
        return - np.sum(Y * np.log(sigmoid(np.dot(X, theta))) + (1 - Y) * np.log(1 - sigmoid(np.dot(X, theta)))) + lmbda * np.sum(np.square(theta))
    elif model == 'Probit':
        return - np.sum(Y * np.log(probit(np.dot(X, theta))) + (1 - Y) * np.log(1 - probit(np.dot(X, theta)))) + lmbda * np.sum(np.square(theta))

def grad_log_loss_glm(theta, X, Y, lmbda, model):
    if model == 'Logistic':
        return - np.dot(Y, X) + np.dot(sigmoid(np.dot(X, theta)), X) + lmbda * theta
    elif model == 'Probit':
        return - np.dot(Y, X) + np.dot(probit(np.dot(X, theta)), X) + lmbda * theta

def hess_log_loss_glm(theta, X, Y, lmbda, model):
    if model == 'Logistic':
        return np.sum([dsigmoid(np.dot(theta, x)) * np.outer(x, x) for x in X], axis=0) + lmbda*np.eye(theta.shape[0])
    elif model == 'Probit':
        return np.sum([dprobit(np.dot(theta, x)) * np.outer(x, x) for x in X], axis=0) + lmbda*np.eye(theta.shape[0])

def probit(x):
    return norm.cdf(x)

def dprobit(x):
    return (1.0 / np.sqrt(2*np.pi)) * np.exp(-x*x/2.0)

def solve_glm_mle(theta_prev, X, Y, lmbda, model):
    # res = minimize(log_loss_glm, theta_prev,\
    #                jac=grad_log_loss_glm, hess=hess_log_loss_glm, \
    #                 args=(X, Y, lmbda, model), method='Newton-CG')
    res = minimize(log_loss_glm, theta_prev, args=(X, Y, lmbda, model))
    # if not res.success:
    #     print(res.message)

    theta_hat, succ_flag = res.x, res.success
    return theta_hat, succ_flag
####################################################