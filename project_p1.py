'''Convex Optimization Project 2016
Written by: Andrew Stier, Farzan Memarian, Sid Desai'''

from __future__ import division
import numpy as np
import random
import numpy.random as rn
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
import sys
from scipy import linalg as LA
import math

ITERATIONS = 50
LAM = 0.1
C = 0.1
TAU = 1000
STEPSIZE = 0.1
EPSILON = 10**(-10)
BATCH_SIZE = 2

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
	gf = np.dot(H,(x - X_STAR))
	return gf

def ogradf(x, indices):
	#Calculates the gradient of the function
	H = np.dot(J,J.T)
	x_sub = x[indices]
	x_star_sub = X_STAR[indices]
	H_sub = H_sub = H[:,indices]
	gf = np.dot(H_sub,(x_sub - x_star_sub))
	return gf

def f(x):
	#calculates the function of x
	H = np.dot(J,J.T)
	fx = 1/2 * np.dot(np.dot((x - X_STAR).T,H),x - X_STAR)
	return fx

def linemin(Tau, t, n):
	#update the stepsize
	return Tau/(Tau + t) * n



def bfgs(x, t, B, n, Tau):
	'''
	Update x based on the BFGS algorithm as listed in algorithm 1
	Steps 3a-3i
	'''


	I = np.diag(np.ones(len(x)))
	gf = gradf(x)

	#Steps a through i of algorithm 1
	p = np.dot(-B, gf)
	n = linemin(Tau, t, n)
	s = n*p
	x_new = x + s
	y = gradf(x_new) - gf
	if t == 0:
		B = np.dot(s.T,y)/np.dot(y.T,y) * I
	ro = 1/np.dot(s.T, y)
	B = np.dot(np.dot((I - ro * np.dot(s, y.T)),B),(I - ro*np.dot(y,s.T))) + ro*np.dot(s,s.T)

	return x_new, B, n

def obfgs(x, t, B, n, Tau):
	I = np.diag(np.ones(len(x)))
	
	#Step 2
	if t == 0:
		B = EPSILON * I

	#Steps 3a - 3h
	indices = random.sample(range(0, len(x)), BATCH_SIZE)
	gf = ogradf(x, indices)

	#Steps a through i of algorithm 1
	p = np.dot(-B, gf)
	n = linemin(Tau, t, n)
	s = n/C*p
	x_new = x + s
	y = ogradf(x_new, indices) - gf + LAM * s
	if t == 0:
		B = np.dot(s.T,y)/np.dot(y.T,y) * I
	ro = 1/np.dot(s.T, y)
	B = np.dot(np.dot((I - ro * np.dot(s, y.T)),B),(I - ro*np.dot(y,s.T))) + C*ro*np.dot(s,s.T)

	return x_new, B, n

def descent(update, x_start, x_star, n, Tau, T=100):
	'''
	This function does descent optimization using the update method of choice.
	The l2 error is recorded each iteration.

	Args:
	update: update method of choice
	x_start: initial guess for x
	x_star: the correct optimal x
	n: the initial stepsize
	Tau: determines how much the stepsize will decrease by each iteration
	T: The number of iterations to perform

	Output:
	x: The optimal x found by the descent algorithm
	error: A list containing the error each iteration
	'''

	x = x_start

	# B starts as the identity matrix
	B = np.diag(np.ones(len(x_start)))

	error = [la.norm(x - x_star)**2]

	for t in xrange(T):
	    
	    x, B, n = update(x, t, B, n, Tau)

	    if (t % 1 == 0) or (t == T - 1):
	        # calculate the error of this iteration
	        error.append(la.norm(x - x_star)**2)

	        assert not np.isnan(error[-1])

	return x, error


def main():
	N = 5 #dimension
	
	x_star = np.ones((N,1)) #The optimal answer
	x_start = np.ones((N,1)) * 3 #The arbitrary point we start from

	#tuning parameters which dictate how the stepsize will change
	Tau = TAU
	n = STEPSIZE

	#optimize using BFGS
	x, errors_1 = descent(bfgs, x_start, x_star, n, Tau, T=ITERATIONS)

	#optimize using online BFGS, AKA stochastic BFGS
	x, errors_2 = descent(obfgs, x_start, x_star, n, Tau, T=ITERATIONS)


	# plot error vs. iteration for both
	plt.clf()
	plt.plot(errors_1, label="BFGS")
	plt.plot(errors_2, label="oBFGS")
	plt.title('Error')
	plt.legend()
	plt.show()
	plt.savefig('Problem1.eps')



if __name__ == "__main__":
    main()