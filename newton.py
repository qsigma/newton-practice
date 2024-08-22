import numpy as np

def func(x):
    return np.cos(x)

def first_order_deriv(x, eps=1e-4):
    """Calculate the first derivative."""
    if not eps > 0:
        eps = 1e-4
    return (func(x+eps)-func(x))/eps

def second_order_deriv(x, eps=1e-4):
    """Calculate the second derivative."""
    if not eps > 0:
        eps = 1e-4
    return (first_order_deriv(x+eps)-first_order_deriv(x))/eps

def newton(x, threshold=0.01):
    """
    Do newton iterations, until convergence.

    Keyword arguments:
    x -- the initial position
    threshold -- the threshold for the changing value to be small enough to terminate the iteration
    """
    if not threshold > 0:
        threshold = 0.01
    first_deriv = first_order_deriv(x)
    second_deriv = second_order_deriv(x)
    while abs(first_deriv/second_deriv) > threshold:
        x = x - first_deriv/second_deriv
        first_deriv = first_order_deriv(x)
        second_deriv = second_order_deriv(x)
    return x

if __name__ == "__main__":
    print(newton(1))
    