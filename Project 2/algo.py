import math # for math.inf

# these 2 libraries for random graph generation
import networkx as nx
import numpy as np

# for measuring time
from time import process_time_ns

# ignores networkx warnings
import warnings
warnings.filterwarnings("ignore")

class GraphNode:
    def __init__(self, vertex: int, weight: int = 0):
        self.vertex = vertex
        self.weight = weight
        self.next = None

class Graph:
    def __init__(self, V: int, adj: list[GraphNode] | list[list[int]]):
        self.V = V
        self.adj = adj

class PQArr:
    def __init__(self, size: int):
        self.size = 0
        self.pos = [0] * size
        self.items = [(0,0)] * size
    
    def insert(self, item: int, priority: int):
        self.pos[item] = self.size
        self.items[self.size] = (item,priority)
        self.size += 1
    
    def update_priority(self, item: int, priority: int):
        self.items[self.pos[item]] = (item, priority)

    def get_min(self) -> int:
        min_item = self.items[0][0]
        minimum = self.items[0][1]
        for x in range(1,self.size):
            if(self.items[x][1] < minimum):
                minimum = self.items[x][1]
                min_item = self.items[x][0]

        return min_item

    def extract_min(self) -> int: 
        min_item = self.get_min()
        min_item_pos = self.pos[min_item]

        last_item = self.items[self.size - 1]
        self.items[min_item_pos] = last_item
        self.pos[last_item[0]] = min_item_pos
        self.size -= 1

        return min_item

class PQHeap:
    def __init__(self, size: int):
        self.size = 0
        self.pos = [0] * size
        self.items = [(None,0)] * (size+1)

    def swap_items(self, a: int, b: int):
        temp = self.items[a]
        temp_pos = self.pos[temp[0].vertex]
        self.pos[temp[0].vertex] = self.pos[self.items[b][0].vertex]
        self.items[a] = self.items[b]
        self.pos[self.items[b][0].vertex] = temp_pos
        self.items[b] = temp

    def fix_up(self, N: int):
        while N > 1 and self.items[N//2][1] > self.items[N][1]:
            self.swap_items(N//2,N)
            N = N // 2

    def fix_down(self, k: int, N: int):
        while 2 * k <= N:
            j = 2 * k
            if j < N and self.items[j][1] > self.items[j+1][1]:
                j += 1
            if self.items[k][1] <= self.items[j][1]:
                break
            self.swap_items(k,j)
            k = j

    def decrease_key(self, item: GraphNode, priority: int):
        self.items[self.pos[item.vertex]] = (item,priority)
        self.fix_up(self.pos[item.vertex])
    
    def insert(self, item: GraphNode, priority: int):
        self.size += 1
        self.pos[item.vertex] = self.size
        self.items[self.size] = (item,priority)
        self.fix_up(self.size)

    def extract_min(self) -> GraphNode:
        self.swap_items(1,self.size)
        min_node = self.items[self.size][0]
        self.fix_down(1,self.size-1)
        self.size -= 1
        return min_node

# converts adjacency matrix to list form
def adj_mat_2_list(g: Graph):
    adj_list = [None] * g.V
    for i in range(0,g.V):
        adj_list[i] = GraphNode(i)
        cur_node = adj_list[i]
        for j in range(0,g.V):
            w = g.adj[i][j]
            if w == 0:
                continue
            
            new_node = GraphNode(j,w)
            cur_node.next = new_node
            cur_node = new_node
    g.adj = adj_list

def dijkstra_a(g : Graph, source : int):
    d = [math.inf] * g.V
    pi = [None] * g.V
    S = [0] * g.V

    d[source] = 0

    pq = PQArr(g.V)
    for x in range(0,g.V):
        pq.insert(x,d[x])

    for x in range(0,g.V-1):
        u = pq.extract_min()
        S[u] = 1

        for v in range(0,g.V):
            w = g.adj[u][v]
            if w == 0:
                continue

            current_distance = d[v]
            new_distance = d[u] + w
            if not S[v] and current_distance > new_distance:
                d[v] = new_distance
                pi[v] = u
                pq.update_priority(v,new_distance)
    
    return (d,pi)

def dijkstra_b(g: Graph, source: GraphNode):
    d = [math.inf] * g.V
    pi = [None] * g.V
    S = [0] * g.V

    d[source.vertex] = 0

    pq = PQHeap(g.V)
    for x in range(0,g.V):
        pq.insert(g.adj[x],d[x])

    while(pq.size > 0):
        u = pq.extract_min()
        S[u.vertex] = 1

        v = g.adj[u.vertex].next
        while(v):
            current_distance = d[v.vertex]
            new_distance = d[u.vertex] + v.weight
            if not S[v.vertex] and current_distance > new_distance:
                d[v.vertex] = new_distance
                pi[v.vertex] = u.vertex
                pq.decrease_key(v,new_distance)
            v = v.next

    return (d,pi)
    
# used to verify our dijkstra is working by comparing the distance array with networkx's
def verify(WG, d):
    GP = nx.from_numpy_matrix(WG)
    length, _ = nx.single_source_dijkstra(GP,0)
    for x in range(0,len(length)):
        if d[x] != length[x]:
            print(d[x], length[x])

seed = 20012001
def run_dijks(n: int, p:int, WG:list[list[int]]=None, E:int=None):
    # if a matrix is not provided, generate one with given n and p
    if WG is None:
        global seed;
        while True:
            G = nx.generators.random_graphs.gnm_random_graph(n, p, seed=seed) # generate random graph with values of n and p
            seed += 1
            if nx.is_connected(G):
                break

        A = nx.adjacency_matrix(G).toarray()
        r = np.random.default_rng(seed*2)
        W = r.integers(n, n*10, size=(n, n))
        WG = np.multiply(W,np.tril(A))
        WG += WG.T
        E = G.number_of_edges()

    g = Graph(n,WG)
    
    t1_start = process_time_ns() 
    dijkstra_a(g,0)
    t1_stop = process_time_ns()

    adj_mat_2_list(g)
    t2_start = process_time_ns() 
    dijkstra_b(g,g.adj[0])
    t2_stop = process_time_ns() 

    log_v = math.log(n,2)
    print(f"{E}\t{int(E * log_v)}\t\t{int((E+n) * log_v)}\t\t{int(t2_stop-t2_start)/1e9:.4g}\t\t{int(t1_stop-t1_start)/1e9:.4g}")

# Makes n x n worst case matrix for Dijkstra and runs both implementations
def worst_case(n: int):
    e = np.zeros(shape=(n,n),dtype='int')
    for x in range(0,n-1):
        e[x,x+1] = 1
        e[x+1,x] = 1
        
    for i in reversed(range(0,n-2)):
        for j in range(i+2,n):
            amount = e[i,i + 1] + e[i + 1,j] + 1
            e[i,j] = amount
            e[j,i] = amount

    run_dijks(n, 0, e, (n * n) - n)
    return e

print("E\tE*log(V)\t(E+V)*log(V)\tList Time\tMatrix Time")

# run the dijkstra algorithm with different values of n and p
for x in range(50000,500001,50000):
    run_dijks(1000,x)

# run the dijkstra algorithm with worst case matrix for different values of n
worst_case(1000)