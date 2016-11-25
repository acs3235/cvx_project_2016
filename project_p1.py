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

def make_J_matrix(n):
	#Makes the matrix J as defined for a simple example function
	J = np.zeros((n,n))

	for i in range(1,n+1):
		for j in range(1,n+1):
			if i % j == 0 or j % i == 0:
				J[i-1][j-1] = 1/(i + j - 1)

	return J

J = make_J_matrix(5)
X_STAR = np.ones((5,1))

def gradf(x):
	#Calculates the gradient of the function
	H = np.dot(J,J.T)
	gf = H * (x - X_STAR)
	return gf

def f(x):
	#calculates the function of x
	H = np.dot(J,J.T)
	fx = 1/2 * np.dot(np.dot((x - X_STAR).T,H),x - X_STAR)
	return fx

def linemin(f,x,p):
	return 0.1



def bfgs(x, t, B):
	I = np.diag(np.ones(len(x)))
	gf = gradf(x)

	#Steps a through i of algorithm 1
	p = np.dot(-B, gf)
	n = linemin(f,x,p)
	s = n*p
	x_new = x + s
	y = gradf(x_new) - gf
	if t == 0:
		B = np.dot(s.T,y)/np.dot(y.T,y) * I
	ro = 1/np.dot(s.T, y)
	B = np.dot(np.dot((I - ro * np.dot(s, y.T)),B),(I - ro*np.dot(y,s.T))) + ro*np.dot(s,s.T)

	return x_new, B

def obfgs(x,t,B):
	return x, B

def descent(update, x_start, x_star, T=100):
    x = x_start
    B = np.diag(np.ones(len(x_start)))

    error = []
    for t in xrange(T):
        
        x, B = update(x, t, B)

        if (t % 1 == 0) or (t == T - 1):
            #calculate the error of this iteration
            error.append(la.norm(x - x_star)**2)

            assert not np.isnan(error[-1])

    return x, error

def main():
	n = 5
	
	x_star = np.ones((n,1))
	x_start = np.ones((n,1)) * 3
   
	x, errors_1 = descent(bfgs, x_start, x_star, T=5)
	x, errors_2 = descent(obfgs, x_start, x_star, T=5)


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