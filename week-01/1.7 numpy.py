import numpy as np
# Creating arrays
np.zeros(10)

np.ones(10)

np.full(10, 2.5)

a = np.array([1, 2, 3, 5, 7, 12])
a
a[2]
a[2] = 10
a[2]

np.arange(3, 10)

np.linspace(0, 100, 11)

# Multi-dimensional arrays
np.zeros((5, 2))

n = np.array( [
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

n[0, 1] = 20

n[:,1]

# Randomly generated arrays
np.random.seed(2)
100 * np.random.rand(5, 2)

np.random.seed(2)
np.random.randn(5, 2)

np.random.seed(2)
np.random.randint(low=0, high=100, size=(5,2))

# Element-wise operations
a = np.arange(5)
a
a + 1
a * 2
b = (10 + (a * 2)) ** 2 / 100
b
a / b + 10

# Comparison operations
a >= 2
a > b
a[a > b]

# Summarizing operations
a
a.min()
a.max()
a.sum()
a.mean()
a.std()