import numpy as np

# Unbounded knapsack problem. O(n^2) time and space. Slightly tweaked from the original problem.
def unbounded_knapsack_n2(w : list[int], p : list[int], C : int, n : int):
    # initialize an empty 2d array filled with zeroes of size [C+1][n+1]
    profit = np.zeros(shape=(C+1,n+1),dtype='int') 

    for c in range(1,C+1):
        for j in range(1,n+1):
            profit[c][j] = profit[c][j-1]
            if w[j] <= c and profit[c][j] < profit[c-w[j]][j] + p[j]:
                    profit[c][j] = profit[c-w[j]][j] + p[j]
    
    return profit

# Unbounded knapsack problem. O(n^2) time and O(n) space. Similar to rod cutting problem.
def unbounded_knapsack_n(w : list[int], p : list[int], C : int, n : int):
    profit = [0] * (C+1)

    for c in range(1,C+1):
        for j in range(1,n+1):
            if w[j] <= c:
                profit[c] = max(profit[c], p[j] + profit[c-w[j]])
    
    return profit

# try out the different implementations and problems
print(unbounded_knapsack_n2([0,4,6,8],[0,7,6,9],14,3))
print(unbounded_knapsack_n([0,4,6,8],[0,7,6,9],14,3))
print(unbounded_knapsack_n2([0,5,6,8],[0,7,6,9],14,3))
print(unbounded_knapsack_n([0,5,6,8],[0,7,6,9],14,3))