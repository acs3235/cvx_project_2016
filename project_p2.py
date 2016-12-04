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
from sklearn.datasets import fetch_rcv1

DEBUG = 1
DATASIZE = 10
N = 3 #dimension

# rcv1 = fetch_rcv1()
# data = rcv1.data
# target = rcv1.target

# X = data
# z = target[:,0]

data = np.loadtxt('Skin_NonSkin.txt')

X = data[:,:3]
z = data[:,3]

print X.shape
print z.shape


ITERATIONS = 4
LAM = 0.1
C = 0.1
TAU = 1000
STEPSIZE = 0.1
EPSILON = 10**(-10)
BATCH_SIZE = 10


if(DEBUG == 1):
	print "debug mode"
	X = X[:DATASIZE,:N]
	z = z[:DATASIZE]


def c(w, xi):
	# xi = xi.todense()
	return 1/(1 + math.exp(-np.dot(xi.T,w)))

def fi(w, xi, zi):
	return zi * math.log(c(w, xi)) + (1 - zi) * math.log(1 - c(w,xi))

def gfi(w,xi,zi):
	return np.asscalar(c(w,xi) - zi)*xi

def f(w):
	ans = 0
	for i, zi in enumerate(z):
		xi = X[i,:].T
		ans = ans + fi(w,xi,zi)
	return np.asscalar(ans)

def gradf(x):
	#Calculates the gradient of the function
	# H = np.dot(J,J.T)
	# gf = np.dot(H,(x - X_STAR))
	gf = 0
	return gf

def ogradf(w, indices):
	#Calculates the batch gradient of the function
	gf = 0
	for i in indices:
		xi = X[i,:].T
		zi = z[i]
		gf = gf + gfi(w,xi,zi)
	return gf

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
	indices = random.sample(range(0, len(z)), BATCH_SIZE)
	gf = ogradf(x, indices)

	#Steps a through i of algorithm 1
	p = np.dot(-B, gf)
	n = linemin(Tau, t, n)
	s = n/C*p
	x_new = x + s
	y = ogradf(x_new, indices) - gf + LAM * s
	if t == 0:
		B = np.asscalar(np.dot(s.T,y)/np.dot(y.T,y)) * I
	ro = 1/np.asscalar(np.dot(s.T, y))
	# print "got here"
	part1 = np.dot((I - ro * np.dot(s, y.T)),B)
	# print "1"
	part2 = (I - ro*np.dot(y,s.T))
	# print "2"
	part3 = C*ro*np.dot(s,s.T)
	# print "3"
	B = np.dot(part1,part2) + part3

	return x_new, B, n

def descent(update, x_start, n, Tau, T=100):
	'''
	This function does descent optimization using the update method of choice.
	The l2 error is recorded each iteration.

	Args:
	update: update method of choice
	x_start: initial guess for x
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

	error = [f(x)]

	for t in xrange(T):
	    
	    x, B, n = update(x, t, B, n, Tau)

	    if (t % 1 == 0) or (t == T - 1):
	        # calculate the error of this iteration
	        error.append(f(x))

	        assert not np.isnan(error[-1])

	return x, error


def main():
	
	# x_star = np.ones((N,1)) #The optimal answer
	x_start = np.ones((N,1)) * 0.5 #The arbitrary point we start from

	#tuning parameters which dictate how the stepsize will change
	Tau = TAU
	n = STEPSIZE

	#optimize using BFGS
	# x, errors_1 = descent(bfgs, x_start, x_star, n, Tau, T=ITERATIONS)

	#optimize using online BFGS, AKA stochastic BFGS
	x, errors_2 = descent(obfgs, x_start, n, Tau, T=ITERATIONS)

	print errors_2[1]

	# plot error vs. iteration for both
	plt.clf()
	# plt.plot(errors_1, label="BFGS")
	plt.plot(errors_2, label="oBFGS")
	plt.title('Error')
	plt.legend()
	plt.show()
	plt.savefig('Problem1.eps')



if __name__ == "__main__":
    main()