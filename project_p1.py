'''Convex Optimization Project 2016
Written by: Andrew Stier, Farzan Memarian, Sid Desai'''

from __future__ import division
import numpy as np


def make_J_matrix(n):
	J = np.zeros((n,n))

	for i in range(1,n+1):
		for j in range(1,n+1):
			if i % j == 0 or j % i == 0:
				J[i-1][j-1] = 1/(i + j - 1)

	return J

def main():
	n = 5
	J = make_J_matrix(n)
	print J


if __name__ == "__main__":
    main()