# profiling.py
"""Python Essentials: Profiling.
Jake Callahan
Section 002
<Date>
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
import math
import time
from numba import jit
from matplotlib import pyplot as plt


# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    #Read in file
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    #Set initial row position
    j = len(data) - 2
    #Add max values from bottom up
    while j >= 0:
        for k in range(len(data[j])):
            data[j][k] = max((data[j][k] + data[j+1][k]), (data[j][k] + data[j+1][k+1]))
        j -= 1

    #return top value
    return data[0][0]


# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    #Initialize list of primes
    primes_list = [2, 3]
    #Initialize first number to check and fix list size
    current = 5
    N -= 2

    #Iterate until list size reaches 0
    while N > 0:
        #Get square root of checked number
        root = math.sqrt(current)
        isprime = True

        for i in primes_list:     # Check for nontrivial divisors.
            if i > root:
                break
            if current % i == 0:
                isprime = False
                break

        #Append the prime numbers and decrement list size
        if isprime:
            primes_list.append(current)
            N -= 1
        #Move to next odd number
        current += 2

    return primes_list

# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    #Subtract x from A, get the norms, and find the minimum
    return np.argmin(np.linalg.norm(A - np.transpose([x]), axis=0))

# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    #Read in name file and sort
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))

    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    #Read in filename and sort
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    #Set constant to change ASCII ord into numerical value
    ord_const = 64
    #Get number of names
    num_names = len(names)
    #Initialize array to store values
    values = np.zeros(num_names)

    #Calculate raw score for each name in list
    for i, name in enumerate(names):
        values[i] = sum((ord(n) - ord_const for n in name))

    #Multiply scores by position in list
    total = values * np.arange(1, num_names + 1)
    return total.sum()

# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    #Base case 1
    F_1 = 1
    yield F_1
    #Base case 2
    F_2 = 1
    yield F_2

    #Generate next fibonacci numbers
    while True:
        F_3 = F_1 + F_2
        F_1 = F_2
        F_2 = F_3
        yield F_3

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    #Get values from generator up to N, return index
    for i, x in enumerate(fibonacci()):
        if len(str(x)) >= N:
            break
    return i + 1


# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    #Get list of numbers up to N
    nums = [2] + [x for x in range(3,N+1,2)]

    #Yield first number, then remove first number and everything it divides
    while len(nums) != 0:
        yield nums[0]
        nums = [num for num in nums if num % nums[0] != 0]

# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

#Decorate with numba for ludicrous speed
@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    #Initialize lists to store times
    numba_times = []
    numpy_times = []
    pure_times = []

    #Compile the numba matrix code
    A = np.random.random(9).reshape((3,3))
    matrix_power_numba(A,1)

    #Set domain
    domain = 2**np.arange(2,8)
    #Compute matrix powers for increasing powers
    for m in domain:
        #Get initial matrix
        A = np.random.random(m**2).reshape((m,m))

        #Get time for numba matrix
        start = time.time()
        matrix_power_numba(A, n)
        end = time.time()

        numba_times.append(end - start)

        #Get time for numba matrix
        start = time.time()
        matrix_power(A, n)
        end = time.time()

        pure_times.append(end - start)

        #Get time for numba matrix
        start = time.time()
        np.linalg.matrix_power(A, n)
        end = time.time()

        numpy_times.append(end - start)

    #Plot times on log log scale
    plt.loglog(domain, pure_times, basex=2, label="Python times")
    plt.loglog(domain, numba_times, basex=2, label="Numba times")
    plt.loglog(domain, numpy_times, basex=2, label="Numpy times")

    plt.xlabel("Power of matrix")
    plt.ylabel("Time to compute")
    plt.legend(loc="upper left")

    plt.show()
