"""
EECS 445 - Introduction to Machine Learning
HW1 Q4 Perceptron Algorithm with Offset
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
from helper import load_data

def all_correct(X: np.array, y: np.array, theta: np.array, b: float):
    """
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        theta: np.array, shape (d,), normal vector of decision boundary
        b: float, offset

    Returns true if the linear classifier specified by theta and b correctly classifies all examples
    """
    for Xi, yi in zip(X,y):
        if yi * (theta @ Xi + b) <= 0:
            return False
    return True
         
    


def perceptron(X: np.array, y: np.array):
    """
    Implements the Perception algorithm for binary linear classification.
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)

    Returns:
        theta: np.array, shape (d,)
        b: float
        alpha: np.array, shape (n,). 
            Misclassification vector, in which the i-th element is has the number of times 
            the i-th point has been misclassified)
    """
    n, d = X.shape
    theta = np.zeros((d,))
    b = 0
    alpha = np.zeros((n,))

    # TODO: Implement the Perceptron algorithm
    while not all_correct(X,y,theta,b):
        for i,(Xi,yi) in enumerate(zip(X,y)):
            if yi * (theta @ Xi + b) <= 0:
                theta += yi * Xi
                b += yi
                alpha[i]+=1
    return theta, b, alpha



def main(fname):
    X, y = load_data(fname)
    theta, b, alpha = perceptron(X, y)

    print("Done!")
    print("============== Classifier ==============")
    print("Theta: ", theta)
    print("b: ", b)

    print("\n")
    print("============== Alpha ===================")
    print("i \t Number of Misclassifications")
    print("========================================")
    for i in range(len(alpha)):
        print(i, "\t\t", alpha[i])
    print("Total Number of Misclassifications: ", np.sum(alpha))
    n, d = X.shape
    thetamath = np.zeros((d,))
    bmath = float(0)
    for ai,yi,Xi in zip(alpha,y,X):
        thetamath += ai * yi * Xi
        bmath += ai * yi
    
    print("Theta math: ", thetamath)
    print("b math: ", bmath)


if __name__ == '__main__':
    main("dataset/q4.csv")
