import numpy as np
import numpy.ma as ma

# BROADCASTING --------------------------------------------------------

a = np.array([1, 2, 3])  # Shape (3, )
b = np.array([4, 5, 6])  # Shape (3, )
c = np.array([4])        # Shape (1, )
d = np.array([[4], [5]]) # Shape (2, 1)

print(a + b) # Element-wise addition. 
print(a + c) # Broadcasting; treats c as [4,4,4].
print(a + d) # Broadcasting; treats a as [[1,2,3],[1,2,3]]

# ADVANCED INDEXING ---------------------------------------------------

v = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(v[0][1])      # first row second column using array indexing.
print(v[0:2])       # indexing using slicing.
print(v[[0,1]])     # first row and second row using adv. indexing.
print(v[:,:])       # dimensional slicing.

print(v[:,1,np.newaxis])     # all rows, 2nd column, make new axis.
print(v[:, [0,2]])           # Every row, 0th and 2nd column.
print(v[[0,2], [0,2]])       # 0th row col and 2nd row and col.

print(v[:, [True, False, True]]) # all rows, 0th & 2nd col.

# SORTING --------------------------------------------------------------

z = np.array([
    [5, 9, 8],
    [3, 4, 1],
    [0, 0, 7]
])

print(np.sort(z, axis = 1))                       # sorts each row.
print(np.sort(z, axis = 0))                       # sorts each column.
print(np.sort(np.sort(z, axis = 0), axis = 1))    # sorts row & column.

# SEARCHING -----------------------------------------------------------

outputs = np.array([1, 2, 3, 4, 5])
print(np.argmax(outputs))  # index of first instance of maximum val.
print(np.argmin(outputs))  # index of first instance of minimum val.
print(np.nonzero(outputs)) # prints all non-zero outputs.

print(np.where(outputs < 3, outputs, 0))

# ITERATING ------------------------------------------------------------

t = np.arange(16).reshape(4, 4)
for elem in np.nditer(t):                   # row iteration.
    print(elem, end = "\n")

for elem in np.nditer(t, order = 'F'):      # col iteration.
    print(elem, end = "\n")

with np.nditer(t, op_flags = ['readwrite']) as it:
    for elem in it:
        elem[...] = elem ** 2

print(t)

# MASKING -------------------------------------------------------------

v = np.array([1, 2, 3, np.nan, np.inf, 4, 5])
masked_arr = ma.masked_array(v, mask = [0, 0, 0, 1, 1, 0, 0])
print(masked_arr.mean())
print(masked_arr.sum())

u = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(ma.masked_greater(u, 4))    # Masks elems greater than 4.
print(ma.masked_inside(u, 4, 7))  # Masks elems between (4,7) inclusive.
print(ma.masked_outside(u, 4, 7)) # Masks elems outiside (4,7) inclusive.

print(ma.masked_where(u % 2 == 0, u))

# VIEWS AND COPIES ----------------------------------------------------

arr = np.array([1, 2, 3, 4, 5, 6])  # initializes array.
new_arr = arr.copy()                # creates a copy.
new_arr = new_arr[0:3]              # slices copy.
new_arr[0] = 10                     # assigns new elem to pos 0 of copy.
print(new_arr)      
print(arr)

# VECTORIZATION -------------------------------------------------------

j = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

def square_if_even(x):
    if x % 2 == 0:
        return x ** 2
    else:
        return x
    
vectorized_func = np.vectorize(square_if_even)
print(vectorized_func(j))

# MATRIX MULTIPLICATION -----------------------------------------------

I = np.array([
    [1, 2],
    [2, 1]
])

J = np.array([
    [3, 1],
    [1, 3]
])

print(I @ J)
print(np.matmul(I, J))

# CUSTOM DATA TYPE ----------------------------------------------------

