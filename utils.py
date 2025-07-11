import numpy as np
from scipy.optimize import minimize , LinearConstraint
from scipy.stats import norm
import torch


def prob_vector(arm , theta):
    '''
    returns a softmax probability vector which also has prob of zero outcome
    '''
    dim = len(arm)
    num_outcomes = int(len(theta) / dim)
    theta_with_zero = np.hstack(([0 for _ in range(dim)] , theta))
    inner_products = np.kron(np.eye(num_outcomes + 1) , arm) @ theta_with_zero
    prob_vector = np.exp(inner_products) / np.sum(np.exp(inner_products))
    return prob_vector 


def gradient_MNL(arm , theta):
    '''
    returns the gradient of the probability vector with resepect to an arm and theta
    ''' 
    z = prob_vector(arm , theta)[1:]    # leaves out the probability for 0 outcome
    return np.diag(z) - np.outer(z , z)



def minimize_loss_torch(theta , X , Y , lmbda , dimension):
    """
    implements a torch based solution to minimizing the multinomial logistic loss
    """

    if len(Y) == 0:
        return theta , True

    class Multinomial_Regression(torch.nn.Module):
        def __init__(self , theta , dimension):
            super().__init__()
            self.num_outcomes = int(len(theta)/dimension)
            self.dimension = dimension
            self.theta = torch.hstack([torch.tensor([0 for _ in range(dimension)]) , torch.tensor(theta)])
            self.linear = torch.nn.Linear(in_features = len(theta) , out_features = 1 , bias = False)
            self.linear.weight.data = self.theta

        def forward(self , arms):
            arms_stacked = torch.vstack([torch.kron(torch.eye(self.num_outcomes+1) , torch.tensor(arms[i])) for i in range(len(arms))])
            return self.linear(arms_stacked).reshape(len(arms) , -1)

    torch.manual_seed(0)
    epochs = 1000
    model = Multinomial_Regression(theta , dimension)
    criterion = torch.nn.CrossEntropyLoss(reduction = "sum" , ignore_index = 0)
    # weight decay sets parameter for L2 reg
    optimizer = torch.optim.SGD(model.parameters() , weight_decay = lmbda/2 , lr = 1e-4)
    
    for e in (range(epochs)):
        y_pred = model.forward(X)
        loss = criterion(y_pred , torch.tensor(Y))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # resetting the first {dimension} components to 0 (corresponding to 0 outcome)
        model.linear.weight.data[:dimension] = torch.tensor([0 for _ in range(dimension)])

    # return the K components corresponding to outcomes 1 to K
    return model.linear.weight.detach().numpy()[dimension:] , True

########################################################################################
# From Sawarni et. al (https://github.com/nirjhar-das/GLBandit_Limited_Adaptivity.git)

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

