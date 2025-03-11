import numpy as np 

# NUMPY ARRAYS --------------------------------------------------------

a = [1, 2, 3, 4, 5]             # Ordinary Python list.
b = np.array([1, 2, 3, 4, 5])   # NumPy arr (written in C)

print(b)        # prints entire arr.
print(b[1])     # prints 1st elem.
print(b[1:])    # prints from first elem to the end.

a_mul = np.array([
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9]
])

print(a_mul)        # prints multi-arr.
print(a_mul[0])     # prints first row of multi-arr.
print(a_mul[0,1])   # prints first row first column of multi-arr.

# ATTRIBUTES ----------------------------------------------------------

a_mul = np.array([
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9]
])
print(a_mul.shape) # shape of multi-arr.
print(a_mul.ndim)  # number of dimensions of multi-arr.
print(a_mul.size)  # product of shape of multi-arr.
print(a_mul.dtype) # datatype of elements.

# DATA TYPES ----------------------------------------------------------

# Recall: NumPy arrs are implemented in C.
# Thus: if a Numpy array hold different types, they will be converted.
# Warning: if different types can't be converted, an error will occur.

# During initialization you can pass a data type to the constructor.
# This will attempt to implicitly convert all elements in the array.
# NumPy's documentation has a full list of the supported data types. 

b_mul = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype = 'float')
print(b_mul)

# FILLING ARRAYS ------------------------------------------------------

c1 = np.full((3, 3, 3), 9) # multi-arr of dimension 3 of all nines.
c2 = np.zeros((3, 3, 3))   # multi-arr of dimension 3 of all zeros.
c3 = np.ones((3, 3, 3))    # multi-arr of dimension 3 of all ones.
c4 = np.empty((3, 3, 3))   # multi-arr of dimension 3 of all default val.

x_values1 = np.arange(0, 100, 5)      # start, stop, step size.
x_values2 = np.linspace(0, 100, 1000) # start, stop, num values.

# NUMPY OBJECTS -------------------------------------------------------

print(np.nan)                       # NaN object.
print(np.inf)                       # Inf object.
print(np.isnan(np.nan))             # Test for NaN.
print(np.isinf(np.inf))             # Test for inf.
print(np.isnan(np.sqrt(-1)))        # Equivalent test for NaN.
print(np.isinf(np.array([10])/0))   # Equivalent test for inf.

# ARRAY MATH OPERATIONS -----------------------------------------------

arr1 = np.array([
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9]
])
print(arr1 + 2)
print(arr1 - 2)
print(arr1 / 2)
print(arr1 * 2)
print(arr1 ** 2)

arr2 = np.array([
    [9, 8, 7],
    [6, 5, 4], 
    [3, 2, 1]
])
print(arr1 + arr2)
print(arr1 - arr2)
print(arr1 / arr2)
print(arr1 * arr2)

# ARRAY METHODS -------------------------------------------------------

a = np.array([1, 2, 3])
a = np.append(a, [4, 5, 6])      # original arr, arr to insert at end.
a = np.insert(a, 0, [-2, -1, 0]) # original arr, pos, arr to insert.
a = np.delete(a, 1, 0)           # arr, index, row/colum : 0/1
print(a)

b = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20]
])
print(b.shape)        
print(b.reshape((5, 4)))        # 5 arrays of 4 elements.
print(b.reshape((20,)))         # 1 array of 20 elements.
print(b.reshape((20, 1)))       # 20 arrays of 1 element.
print(b.reshape((2, 2, 5)))     # 2 tensors, 2 arrays of 5 element.
print(b.flatten())              # Gives one dimensional copy.
print(b.ravel())                # Gives one dimensional version.

# ARRAY STURCTURING METHODS -------------------------------------------

v = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(v.T)

b1 = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

b2 = np.array([
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20]
])

b3 = np.concatenate((b1, b2), axis = 0) # concatenate rows.
b4 = np.concatenate((b1, b2), axis = 1) # concatenate columns.
b5 = np.stack((b1, b2), axis = 1)       # creates a new dimension.
print(b3)
print(b4)
print(b5)

v = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
])

v1 = np.split(v, 2, axis = 0)          # split two times row wise.
v2 = np.split(v, 2, axis = 1)          # split two times column wise.
print(v1)
print(v2)

# AGGREGATING METHODS -------------------------------------------------
v = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
])
print(v.min())
print(v.max())
print(v.std())
print(v.sum())
print(v.mean())
print(np.median(v))

# NUMPY RANDOM --------------------------------------------------------

number = np.random.randint(0, 100)   
numbers1 = np.random.randint(0, 100, size = (3, 3))
numbers2 = np.random.binomial(10, p = 0.4, size = (3, 3))
numbers3 = np.random.normal(loc = 100, scale = 10, size = (3, 3))
numbers4 = np.random.choice([10, 20, 30], size = (3, 3))
print(numbers1)
print(numbers2)
print(numbers3)
print(numbers4)