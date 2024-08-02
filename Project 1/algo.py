from numpy.random import default_rng

# Generate datasets. Size of each dataset is 1000, 10000, 100000, 1000000, 10000000
RANDINT = 20012001
datasets = []
datasize = 1000
for x in range(0,5):
    r = default_rng(RANDINT+x)
    datasets.append(r.integers(low=1, high=10000000, size=(datasize)).tolist())
    datasize *= 10

COMPARISONS = None
# Compare function for sorting. Notes down the number of comparisons made
def compare(a: int, b: int) -> int:
    global COMPARISONS;
    COMPARISONS += 1
    if a == b:
        return 0
    elif a > b:
        return 1
    else:
        return -1

# Merge function for merge sort
def merge(a:list[int], n:int, m:int) -> None:
    mid = (n + m) // 2
    if m - n <= 0:
        return
    
    left_list = a[n:mid+1]
    right_list = a[mid+1:m+1]
    l1 = len(left_list)
    l2 = len(right_list)

    l = 0
    r = 0
    i = n

    # while left and right list has elements
    while l < l1 and r < l2:
        cmp = compare(left_list[l],right_list[r])
        if cmp <= 0:
            a[i] = left_list[l]
            i, l = i+1, l+1
        if cmp >= 0:
            a[i] = right_list[r]
            i, r = i+1, r+1
        
    if l < l1:
        a[i:i+l1-l] = left_list[l:]

    if r < l2:
        a[i:i+l2-r] = right_list[r:]

# Merge sort function
def merge_sort(a:list[int], n:int, m:int) -> None:
    if(m-n < 1):
        return
    
    if m-n > 1:
        mid = (n + m) // 2
        # merge sort left and right list recursively
        merge_sort(a,n,mid)
        merge_sort(a,mid+1,m)
    
    merge(a,n,m)

# Insertion sort function
def insertion_sort(a:list[int], n:int, m:int) -> None:
    if m - n < 1:
        return

    for i in range(n+1,m+1):
        for j in range(i,n,-1):
            cmp = compare(a[j],a[j-1])
            if cmp < 0:
                a[j], a[j-1] = a[j-1], a[j]
            else:
                break

# Hybrid sort function
def hybrid_sort(a:list[int], n:int, m:int, s:int) -> None:
    if m - n < s:
        insertion_sort(a,n,m)
    else:
        mid = (n + m) // 2
        hybrid_sort(a,n,mid,s)
        hybrid_sort(a,mid+1,m,s)
        merge(a,n,m)

# Run merge sort on given dataset index 
def run_merge(x:int) -> None:
    global COMPARISONS;
    COMPARISONS = 0
    ds = datasets[x].copy()
    merge_sort(ds,0,len(ds)-1)
    print(f"Merge Sort ({len(ds)}): {COMPARISONS}")

# Run hybrid sort on given dataset index and S value
def run_hybrid(x:int, s:int) -> None:
    global COMPARISONS;
    global S;
    COMPARISONS = 0
    S = s
    ds = datasets[x].copy()
    hybrid_sort(ds,0,len(ds)-1,s)
    print(f"Hybrid Sort ({len(ds)}) with S={s}: {COMPARISONS}")

# Try out merge sort and hybrid sort on different dataset sizes and S values
print("Number of Comparisons")
run_merge(2)
run_hybrid(2, 3)
run_hybrid(2, 5)
run_hybrid(2, 10)