"""
EECS 445 - Introduction to Maching Learning
HW1 Q6 Linear Regression Optimization Methods
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
from helper import load_data
import time

def calculate_squared_loss(X, y, theta):
    """
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        theta: np.array, shape (d,). Specifies an (d-1)^th degree polynomial
        
    Returns:
        The squared loss for the given data and parameters
    """
    total = 0
    for Xi,yi in zip(X,y):
        total += ((yi - theta @ Xi)**2)/2
    
    total /= len(X)
    return total

def ls_gradient_descent(X, y, learning_rate=0):
    """
    Implement the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the following stopping criteria together: number of iterations >= 1e6 or |new_loss − prev_loss| <= 1e−10

    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
    
    Returns:
        theta: np.array, shape (d,)
    """
    n, d = X.shape
    theta = np.zeros(d)

    eps = 1e-10
    max_iter = 1e6
    n_iter = 0

    step = learning_rate
    prev_loss = np.inf
    new_loss = calculate_squared_loss(X, y, theta)



    while n_iter < max_iter and abs(new_loss - prev_loss) > eps: # Implement the correct stopping criteria
        n_iter += 1
        
        grad = -X.T @ (y - X @ theta)

        theta = theta - (step * grad)

        prev_loss = new_loss
        new_loss = calculate_squared_loss(X, y, theta)

    print("Learning rate:", learning_rate, "\t\t\tNum iterations:", n_iter)
    return theta



def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Implement the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the following stopping criteria together: number of iterations >= 1e6 or |new_loss − prev_loss| <= 1e−10
    
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        learning_rate: the learning rate for the algorithm
    
    Returns:
        theta: np.array, shape (d,)
    """
    n, d = X.shape
    theta = np.zeros(d)
    adaptive = (learning_rate == 'adaptive')

    eps = 1e-10
    max_iter = 1e6
    n_iter = 0
    epochs = 0

    step = learning_rate
    prev_loss = np.inf
    new_loss = calculate_squared_loss(X, y, theta)

    while n_iter < max_iter and abs(new_loss - prev_loss) > eps:
        epochs += 1
        if adaptive:
            step = 0.1/(1 + 0.1 * n_iter)

        for xt, yt in zip(X, y):
            scale = (yt - theta @ xt) * xt
            if scale.any():
                theta += step * scale
            n_iter += 1

        prev_loss = new_loss
        new_loss = calculate_squared_loss(X, y, theta)

    print("Learning rate:", learning_rate, "\t\t\tNum iterations:", n_iter)
    return theta

def ls_closed_form_optimization(X, y):
    """
    Implement the closed form solution for least squares regression.

    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)

    Returns:
        theta: np.array, shape (d,)
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y

def main(fname_train):
    # TODO: This function should contain all the code you implement to complete question 6.

    X_train, y_train = load_data(fname_train)
    
    # Appending a column of constant ones to the X_train matrix to make X_train the same dimensions as theta.
    # The term multiplied by theta_0 is x^0 = 1 (theta_0 is a constant), which is why the column contains only ones.
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    learning_rate = 0.1
    for i in range(4):
        start_time = time.process_time()
        print("GD: ", ls_gradient_descent(X_train, y_train, learning_rate))
        end_time = time.process_time()
        print(end_time - start_time)
        start_time = time.process_time()
        print("SGD: ", ls_stochastic_gradient_descent(X_train, y_train, learning_rate))
        end_time = time.process_time()
        print(end_time - start_time)
        learning_rate /= 10

    start_time = time.process_time()
    print("SGD ADP:", ls_stochastic_gradient_descent(X_train, y_train, 'adaptive'))
    end_time = time.process_time()
    print(end_time - start_time)
    start_time = time.process_time()
    print("Closed:", ls_closed_form_optimization(X_train, y_train))
    end_time = time.process_time()
    print(end_time - start_time)
    print("Done!")

if __name__ == '__main__':
    main("dataset/q6.csv")
