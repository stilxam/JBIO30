from sympy import *
from functools import partial
import numpy as np
# declare your variables in the form
x = Symbol("x")
y = Symbol("y")
dct = {x: None, y: None} # all values to none, keys to variables declared above

# change this function to meet your needs
function = (3-x*y)**2

# if a custom step function is needed, make it below and
# pass it after prefilling it with functools partial

def etasqr(itr, c=0.1):
    return c/np.sqrt(itr+1)


def grad(function, wrt):
    return function.diff(wrt)


def gradientDesc(function, eta, gradient, init, dct, maxiter=100):
    difflist = list(dct.keys())
    xs = np.empty((maxiter+1, len(init)))
    xs[0] = np.array(init)

    for j in range(maxiter):
        n = 0
        for key in dct.keys():
            dct[key] = xs[j][n].copy()
            n += 1
        for k in range(len(init)):
            xs[j+1][k] = xs[j][k].copy() - eta(itr=k)*gradient(function, wrt=difflist[k]).evalf(subs=dct)
    return xs


init = [1, 1]
eta = partial(etasqr, c=0.01)

test = gradientDesc(function, eta, grad, init, dct, maxiter=10)
#%%
test


#%%
