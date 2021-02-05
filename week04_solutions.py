# Exercise 1

import numpy as np

def det_2(A):
    '''
    Computes the determinant of a 2x2 matrix.
    '''
    d = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return d


# Testing: comparing to the output of NumPy's function
A = np.random.random([2, 2])
print(det_2(A))
print(np.linalg.det(A))


# ---
# Exercise 2

def minor(A, i, j):
    '''
    Return a copy of the matrix A with row i and column j removed.
    '''
    # Extract slices of A before and after row i, concatenate them
    A = np.concatenate((A[:i, :], A[i + 1:, :]), axis=0)
    
    # Extract slices of A before and after column j, concatenate them
    A = np.concatenate((A[:, :j], A[:, j + 1:]), axis=1)
    
    return A


def det_ce(A):
    '''
    Computes the determinant of A recursively using cofactor expansion.
    '''
    # Find the size of A
    N = A.shape[0]

    # Terminating case: size 2x2
    if N == 2:
        d = det_2(A)

    # Recursive case: use cofactor expansion
    else:
        d = 0
        # Use a loop to add all terms in the sum
        for j in range(N):
            d += ((-1) ** (j + 1)) * A[0, j] * det_ce(minor(A, 0, j))
    return d


# Testing: comparing to the output of NumPy's function
A = np.random.random([4, 4])
print(det_ce(A))
print(np.linalg.det(A))


# ---
# Exercise 3
import time

# Use a loop to time and display results for several values of N
for N in range(5, 11):
    A = np.random.random([N, N])

    t0 = time.time()
    det_ce(A)
    t1 = time.time()

    print(f"N = {N:d}: {t1 - t0:.6f} seconds")

# Cofactor expansion becomes rapidly impractical! Increasing the size of the input matrix
# from NxN to (N+1)x(N+1) increases the number of determinants that need to be computed
# by a factor of N+1; we say that the computational cost grows with N! (factorial of the size of the matrix).

# Extrapolating the computational cost of this code, computing the
# determinant of a 24x24 matrix would be expected to take many billions of years.
# The NumPy function can compute this very quickly:

N = 24
A = np.random.random([N, N])

t0 = time.time()
np.linalg.det(A)
t1 = time.time()

print(f"N = {N:d}: {t1 - t0:.6f} seconds")
