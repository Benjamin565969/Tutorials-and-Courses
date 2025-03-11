import numpy as np 

# Ordinary Python list.
a = [1, 2, 3, 4, 5]

# NumPy arrays are written in C and are optimized for linear algebra.
b = np.array([1, 2, 3, 4, 5]) 

# NumPy arrays can still be accessed and sliced like Python lists.
print(b)
print(b[1])
print(b[1:])
