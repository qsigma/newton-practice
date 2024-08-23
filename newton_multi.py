import numpy as np

def func(x):
    output = (x[0])**2 + (x[1] - 1)**2 + (x[2])**2
    return output

def first_partial(x, i, eps=1e-6):
    """
    Calculate the first partial of func w.r.t. given index

    x -- the position to take the partial
    i -- the given index; no bigger than len(x)
    eps -- the step length for partial
    """
    x0 = x
    x1 = x.copy()
    x1[i] += eps
    return (func(x1)-func(x0))/eps

def second_partial(x, i, j, eps=1e-6):
    """
    Calculate the second partial of func w.r.t. two given indices.
    """
    x0 = x
    x1 = x.copy()
    x1[j] += eps
    partial_i_0 = first_partial(x0, i, eps)
    partial_i_1 = first_partial(x1, i, eps)
    return (partial_i_1 - partial_i_0)/eps

def grad(x, eps=1e-6):
    """Calculate the gradient."""
    array = np.zeros(len(x))
    for i in range(len(x)):
        array[i] = first_partial(x, i, eps)
    return array

def hessian_inv(x, eps=1e-6):
    """Calculate the inv of Hessian matrix of given func and x."""
    matrix = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            matrix[i][j] = second_partial(x, i, j, eps)
    return np.linalg.inv(matrix)

def optimize(x, eps=1e-4, tol=1e-4):
    """x_{t+1} = x_t - H(x_t)^{-1} nabla f(x_t)"""
    while np.linalg.norm(np.dot(hessian_inv(x, eps), grad(x,eps))) > tol:
        x = x - np.dot(hessian_inv(x, eps), grad(x,eps))
    return x

if __name__ == "__main__":
    print(optimize([1,1,1]))