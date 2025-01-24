import numpy as np
from barycentric_spanner import BarycentricSpanner

def information_matrix_set(X , distribution_X):
    '''
    returns the information matrix for the given distribution over X
    '''
    assert len(distribution_X) == len(X)
    soln = np.zeros((len(X[0]) , len(X[0])))
    for p,x in zip(distribution_X , X):
        soln += p * np.outer(x , x)
    return soln

def information_matrix_distibution(D , distribution_D):
    '''
    returns the information matrix for a given distribution over D which is a distribution over X
    '''
    assert distribution_D.shape[0] == D.shape[0]
    soln = information_matrix_set(D[0] , distribution_D[0])
    for d_X , X in enumerate(zip(D[1:] , distribution_D[1:])):
        soln += information_matrix_set(X , d_X)
    return soln

def weighted_norm(x , A):
    return np.sqrt(np.dot(x , np.dot(A , x)))

def g_optimal_design(X , algorithm = "d_optimal" , BS = None):
    '''
    returns the g optimal design for a given set of points
    '''
    dim = len(X[0])
    if algorithm == "barycentric_spanner":
        assert BS is not None
        spanning_set =  BarycentricSpanner(dim , X , BS).spanning_set
        assert spanning_set.shape[0] == dim
        distribution = [0 for _ in range(len(X))]
        for i in range(len(X)):
            if X[i] in spanning_set:
                distribution[i] = 1
        return [i/dim for i in distribution]
    
    elif algorithm == "d_optimal":
        from D_optimal import D_Optimal
        spanning_set = D_Optimal(dim , X)
        distribution = spanning_set.d_optimal_policy
        return distribution
    
    else:
        assert False

def sample_softmax(X , M , alpha):
    '''
    samples an arm from X with a alpha-softmax distribution on the weighted norms with M<
    '''
    weights = [weighted_norm(x , M)**2 for x in X]
    prob_dist_unnormalized = np.array(weights) ** alpha
    prob_dist_normalized = prob_dist_unnormalized / np.sum(prob_dist_unnormalized)
    return X[np.random.choice(len(X) , p = prob_dist_normalized)]

def prob_vector(arm , theta):
    '''
    returns a softmax probability vector with an additional 1 (=exp(0)) in the denominator to account for no action chosen
    '''
    theta_temp = np.reshape(theta , (-1 , arm.shape[0]))
    inner_products = theta_temp @ arm
    # adding the 0 for no action chosen
    inner_products = np.hstack([[0] , inner_products])
    return (np.exp(inner_products) / np.sum(np.exp(inner_products)))[1:]

def gradient_MNL(arm , theta):
    '''
    returns the gradient of the probability vector with resepect to an arm and theta
    ''' 
    z = prob_vector(arm , theta)
    return np.diag(z) - np.outer(z , z)


def log_loss(theta , arms , outcomes , lamda):
    loss = lamda/2 * np.linalg.norm(lamda)**2
    for arm , outcome in zip(arms , outcomes):
        if outcome == 0:
            continue
        loss -= np.log(prob_vector(arm , theta)[outcome-1])
    return loss