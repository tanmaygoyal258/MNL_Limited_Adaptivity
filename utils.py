import numpy as np
from barycentric_spanner import BarycentricSpanner

def information_matrix_set(X , distribution_X):
    '''
    returns the information matrix for the given distribution over X
    '''
    assert distribution_X.shape[0] == X.shape[0]
    soln = np.zeros((X.shape[0] , X.shape[0]))
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

def g_optimal_design(params , X , algorithm = "barycentric_spanner" , BS = None):
    '''
    returns the g optimal design for a given set of points
    '''

    if algorithm == "barycentric_spanner":
        assert BS is not None
        spanning_set =  BarycentricSpanner(params , X , BS).spanning_set
        assert spanning_set.shape[0] == d
        # a uniform distribution over the spanning set gives a C\sqrt{d} optimal design
        d = X[0].shape
        distribution = [0 for _ in range(len(X))]
        for i in range(len(X)):
            if X[i] in spanning_set:
                distribution[i] = 1
        return distribution / d
    else:
        assert False , "yet to implement other algorithms"

def sample_softmax(X , M , alpha):
    '''
    samples an arm from X with a alpha-softmax distribution on the weighted norms with M<
    '''
    weights = [weighted_norm(x , M)**2 for x in X]
    prob_dist_unnormalized = np.array(weights) ** alpha
    prob_dist_normalized = prob_dist_unnormalized / np.sum(prob_dist_unnormalized)
    return X[np.random.choice(len(X) , p = prob_dist_normalized)]