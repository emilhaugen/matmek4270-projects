import numpy as np


def mesh_function(f, t):
    return f(t) 

def func(t):
    tol = 1e-8
    return np.exp(-t) * (0+tol <= t)  * (t <= 3 + tol) + np.exp(-3*t) * (3 + tol < t) * (t <= 4 + tol)

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
