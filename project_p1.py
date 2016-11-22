'''Convex Optimization Project 2016
Written by: Andrew Stier, Farzan Memarian, Sid Desai'''

from __future__ import division
import numpy as np
import numpy.random as rn
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
import sys
from scipy import linalg as LA
import math

def bfgs(x,J):
	return x

def obfgs(x,J):
	return x

def make_J_matrix(n):
	J = np.zeros((n,n))

	for i in range(1,n+1):
		for j in range(1,n+1):
			if i % j == 0 or j % i == 0:
				J[i-1][j-1] = 1/(i + j - 1)

	return J

def descent(update, J, x_start, x_star, T=100):
    x = x_start
    error = []
    for t in xrange(T):
        
        x = update(x, J)

        if (t % 1 == 0) or (t == T - 1):
            #calculate the error of this iteration
            error.append(la.norm(x - x_star)**2)

            assert not np.isnan(error[-1])

    return x, error

def main():
	n = 5
	J = make_J_matrix(n)
	
	x_star = np.ones((5,1))
	x_start = np.ones((5,1))
   
	x, errors_1 = descent(bfgs, J, x_start, x_star, T=20)
	x, errors_2 = descent(obfgs, J, x_start, x_star, T=20)


	# plot results
	plt.clf()
	plt.plot(errors_1, label="BFGS")
	plt.plot(errors_2, label="oBFGS")
	plt.title('Error')
	plt.legend()
	plt.show()
	plt.savefig('Problem1.eps')



if __name__ == "__main__":
    main()