"""
EECS 445 - Introduction to Machine Learning
HW1 Q7 Logistic Regression
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
from helper import load_data

def sigmoid(z):
    """ 
    Implements the sigmoid function..
    Args:
        z: A scalar or numpy array of any size
    """
    # TODO: Implement the sigmoid function
    pass 

def logistic_stochastic_gradient_descent(X, y, lr=0.0001):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for logistic regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of epochs == 10,000
    
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        lr: the learning rate for the algorithm
    
    Returns:
        theta: np.array, shape (d+1,) including the offset term.
    """
    n, d = X.shape
    theta = np.zeros(d + 1)
    
    # TODO: Implement SGD. Train for 10,000 epochs
    return theta

def main(fname):
    X, y = load_data(fname)
    theta_SGD = logistic_stochastic_gradient_descent(X, y)
    print("SGD Theta: ", theta_SGD)


if __name__ == '__main__':
    main('dataset/q7.csv')
