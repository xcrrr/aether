#!/usr/bin/env python3
"""Generate Aether Omega training data from Python source code.

Data sources (in priority order):
  1. Python stdlib       — inspect.getsource() on 50 stdlib modules   trust_score=95
  2. Python builtins     — generated usage patterns for builtins        trust_score=90
  3. Algorithmic patterns — 50+ canonical algorithm implementations     trust_score=92
  4. Augmentation        — variable-rename / style variants of 1-3     trust_score=original*0.85

Output format (JSONL):
  {"input_ids": [int, ...], "labels": [int, ...],
   "trust_score": float, "source": "stdlib|builtin|algo|augmented"}

Labels = input_ids shifted left by one (next-token prediction):
  full_tokens = [BOS, t1, t2, ..., tN, EOS]
  input_ids   = full_tokens[:-1]   (length N+1)
  labels      = full_tokens[1:]    (length N+1, last token EOS)

CLI:
  python generate_data.py \\
      --output data/aether_train.jsonl \\
      --tokenizer omega_tokenizer.json \\
      --n-samples 60000 \\
      --max-seq-len 512 \\
      --seed 42

  python generate_data.py --train-tokenizer \\
      --output data/aether_train.jsonl \\
      --tokenizer omega_tokenizer.json \\
      --n-samples 60000
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import math
import random
import re
import sys
import textwrap
from pathlib import Path

# ── Target stdlib modules ────────────────────────────────────────────────────

STDLIB_MODULES: list[str] = [
    "os", "os.path", "sys", "math", "random", "json", "re", "collections",
    "itertools", "functools", "pathlib", "string", "datetime", "time",
    "hashlib", "io", "copy", "typing", "abc", "dataclasses", "enum",
    "textwrap", "shutil", "threading", "queue", "heapq", "bisect",
    "array", "struct", "base64", "urllib.parse", "argparse", "logging",
    "contextlib", "warnings", "traceback", "inspect", "ast", "dis",
    "tokenize", "zipfile", "tarfile", "csv", "configparser",
    "socket", "ssl", "email.utils",
]


# ── Source 3: Algorithmic patterns ───────────────────────────────────────────

ALGO_PATTERNS: list[str] = [

    # ── Sorting ──────────────────────────────────────────────────────────────
    '''\
def quicksort(arr: list) -> list:
    """Sort a list in place using the quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left   = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right  = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
''',

    '''\
def merge_sort(arr: list) -> list:
    """Merge sort — O(n log n), stable, divide and conquer."""
    if len(arr) <= 1:
        return arr
    mid   = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left: list, right: list) -> list:
    result: list = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
''',

    '''\
def heap_sort(arr: list) -> list:
    """Heap sort using Python's heapq module — O(n log n)."""
    import heapq
    h = arr[:]
    heapq.heapify(h)
    return [heapq.heappop(h) for _ in range(len(h))]
''',

    '''\
def counting_sort(arr: list[int], max_val: int) -> list[int]:
    """Counting sort for non-negative integers — O(n + k)."""
    count = [0] * (max_val + 1)
    for x in arr:
        count[x] += 1
    result: list[int] = []
    for val, freq in enumerate(count):
        result.extend([val] * freq)
    return result
''',

    '''\
def radix_sort(arr: list[int]) -> list[int]:
    """LSD radix sort for non-negative integers — O(d * n)."""
    if not arr:
        return arr
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        arr = _counting_sort_by_digit(arr, exp)
        exp *= 10
    return arr

def _counting_sort_by_digit(arr: list[int], exp: int) -> list[int]:
    n = len(arr)
    output  = [0] * n
    count   = [0] * 10
    for i in arr:
        idx = (i // exp) % 10
        count[idx] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        idx = (arr[i] // exp) % 10
        output[count[idx] - 1] = arr[i]
        count[idx] -= 1
    return output
''',

    # ── Search ───────────────────────────────────────────────────────────────
    '''\
def binary_search(arr: list, target) -> int:
    """Binary search on a sorted list. Returns index or -1."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',

    '''\
def bfs(graph: dict, start) -> list:
    """Breadth-first search — returns visited nodes in BFS order."""
    from collections import deque
    visited = set()
    queue   = deque([start])
    order   = []
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for nbr in graph.get(node, []):
            if nbr not in visited:
                queue.append(nbr)
    return order
''',

    '''\
def dfs(graph: dict, start, visited: set | None = None) -> list:
    """Depth-first search — returns visited nodes in DFS order."""
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]
    for nbr in graph.get(start, []):
        if nbr not in visited:
            result.extend(dfs(graph, nbr, visited))
    return result
''',

    '''\
def dijkstra(graph: dict[str, list[tuple[str, float]]], src: str) -> dict:
    """Dijkstra shortest-path from src. graph[u] = [(v, weight), ...]."""
    import heapq
    dist  = {node: math.inf for node in graph}
    dist[src] = 0.0
    pq    = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
''',

    '''\
def a_star(graph: dict, src, dst, h) -> list:
    """A* search. h(node) is the heuristic function."""
    import heapq
    open_set = [(h(src), 0, src, [src])]
    visited  = set()
    while open_set:
        _, g, node, path = heapq.heappop(open_set)
        if node == dst:
            return path
        if node in visited:
            continue
        visited.add(node)
        for nbr, cost in graph.get(node, []):
            if nbr not in visited:
                ng = g + cost
                heapq.heappush(open_set, (ng + h(nbr), ng, nbr, path + [nbr]))
    return []
''',

    # ── Data structures ───────────────────────────────────────────────────────
    '''\
class Stack:
    """LIFO stack backed by a list."""
    def __init__(self) -> None:
        self._data: list = []
    def push(self, item) -> None:
        self._data.append(item)
    def pop(self):
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()
    def peek(self):
        if not self._data:
            raise IndexError("peek on empty stack")
        return self._data[-1]
    def __len__(self) -> int:
        return len(self._data)
    def __bool__(self) -> bool:
        return bool(self._data)
''',

    '''\
class Queue:
    """FIFO queue backed by collections.deque."""
    from collections import deque
    def __init__(self) -> None:
        self._data: "Queue.deque" = Queue.deque()
    def enqueue(self, item) -> None:
        self._data.append(item)
    def dequeue(self):
        if not self._data:
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()
    def __len__(self) -> int:
        return len(self._data)
''',

    '''\
class LinkedList:
    """Singly-linked list with O(1) prepend and O(n) append."""
    class _Node:
        __slots__ = ("val", "next")
        def __init__(self, val, next=None):
            self.val  = val
            self.next = next

    def __init__(self) -> None:
        self.head = None
        self._len  = 0

    def prepend(self, val) -> None:
        self.head = LinkedList._Node(val, self.head)
        self._len += 1

    def append(self, val) -> None:
        node = LinkedList._Node(val)
        if not self.head:
            self.head = node
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = node
        self._len += 1

    def to_list(self) -> list:
        result, cur = [], self.head
        while cur:
            result.append(cur.val)
            cur = cur.next
        return result

    def __len__(self) -> int:
        return self._len
''',

    '''\
class Trie:
    """Prefix trie for string lookup and autocomplete."""
    def __init__(self) -> None:
        self._children: dict = {}
        self._end: bool = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            if ch not in node._children:
                node._children[ch] = Trie()
            node = node._children[ch]
        node._end = True

    def search(self, word: str) -> bool:
        node = self
        for ch in word:
            if ch not in node._children:
                return False
            node = node._children[ch]
        return node._end

    def starts_with(self, prefix: str) -> bool:
        node = self
        for ch in prefix:
            if ch not in node._children:
                return False
            node = node._children[ch]
        return True
''',

    '''\
class MinHeap:
    """Min-heap implemented from scratch."""
    def __init__(self) -> None:
        self._data: list = []

    def push(self, val) -> None:
        self._data.append(val)
        self._sift_up(len(self._data) - 1)

    def pop(self):
        if not self._data:
            raise IndexError("pop from empty heap")
        self._data[0], self._data[-1] = self._data[-1], self._data[0]
        val = self._data.pop()
        if self._data:
            self._sift_down(0)
        return val

    def peek(self):
        return self._data[0]

    def _sift_up(self, i: int) -> None:
        while i > 0:
            parent = (i - 1) // 2
            if self._data[i] < self._data[parent]:
                self._data[i], self._data[parent] = self._data[parent], self._data[i]
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        n = len(self._data)
        while True:
            smallest = i
            for child in (2 * i + 1, 2 * i + 2):
                if child < n and self._data[child] < self._data[smallest]:
                    smallest = child
            if smallest == i:
                break
            self._data[i], self._data[smallest] = self._data[smallest], self._data[i]
            i = smallest

    def __len__(self) -> int:
        return len(self._data)
''',

    '''\
class BloomFilter:
    """Space-efficient probabilistic set membership test."""
    def __init__(self, capacity: int, error_rate: float = 0.01) -> None:
        m = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        k = int(m / capacity * math.log(2))
        self._m: int = max(m, 1)
        self._k: int = max(k, 1)
        self._bits: list[bool] = [False] * self._m

    def _hashes(self, item: str) -> list[int]:
        import hashlib
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    def add(self, item: str) -> None:
        for pos in self._hashes(item):
            self._bits[pos] = True

    def __contains__(self, item: str) -> bool:
        return all(self._bits[pos] for pos in self._hashes(item))
''',

    '''\
class UnionFind:
    """Disjoint-set union with path compression and union by rank."""
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank   = [0] * n

    def find(self, x: int) -> int:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)
''',

    # ── Dynamic programming ────────────────────────────────────────────────────
    '''\
def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """0/1 knapsack: max value with total weight ≤ capacity."""
    n  = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if w <= c:
                dp[i][c] = max(dp[i][c], dp[i - 1][c - w] + v)
    return dp[n][capacity]
''',

    '''\
def lcs(a: str, b: str) -> str:
    """Longest Common Subsequence of two strings."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # Backtrack
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            result.append(a[i - 1]); i -= 1; j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(result))
''',

    '''\
def edit_distance(s: str, t: str) -> int:
    """Levenshtein edit distance between strings s and t."""
    m, n = len(s), len(t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]
''',

    '''\
def coin_change(coins: list[int], amount: int) -> int:
    """Minimum number of coins to make amount; -1 if impossible."""
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1
''',

    '''\
def longest_increasing_subsequence(nums: list[int]) -> int:
    """LIS length in O(n log n) using patience sorting."""
    import bisect
    tails: list[int] = []
    for x in nums:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)
''',

    '''\
def matrix_chain_order(dims: list[int]) -> int:
    """Minimum scalar multiplications for a chain of matrices.
    dims[i-1] x dims[i] is the dimension of matrix i.
    """
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float("inf")
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][n - 1]
''',

    # ── Graph algorithms ──────────────────────────────────────────────────────
    '''\
def topological_sort(graph: dict) -> list:
    """Topological sort using Kahn's algorithm (BFS-based)."""
    from collections import deque
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] = in_degree.get(v, 0) + 1
    queue = deque(u for u, d in in_degree.items() if d == 0)
    result: list = []
    while queue:
        u = queue.popleft()
        result.append(u)
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(result) != len(graph):
        raise ValueError("Graph has a cycle — topological sort undefined")
    return result
''',

    '''\
def kosaraju_scc(graph: dict) -> list[list]:
    """Kosaraju's algorithm for strongly connected components."""
    order: list = []
    visited: set = set()

    def dfs1(u):
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                dfs1(v)
        order.append(u)

    for node in list(graph):
        if node not in visited:
            dfs1(node)

    # Build reverse graph
    rev: dict = {u: [] for u in graph}
    for u in graph:
        for v in graph[u]:
            rev.setdefault(v, []).append(u)

    visited.clear()
    components: list[list] = []

    def dfs2(u, comp):
        visited.add(u)
        comp.append(u)
        for v in rev.get(u, []):
            if v not in visited:
                dfs2(v, comp)

    for node in reversed(order):
        if node not in visited:
            comp: list = []
            dfs2(node, comp)
            components.append(comp)

    return components
''',

    '''\
def kruskal_mst(n: int, edges: list[tuple[float, int, int]]) -> list:
    """Kruskal's minimum spanning tree. edges = [(weight, u, v), ...]."""
    import heapq
    edges = sorted(edges)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    mst: list = []
    for w, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((w, u, v))
            if len(mst) == n - 1:
                break
    return mst
''',

    '''\
def ford_fulkerson(graph: list[list[int]], source: int, sink: int) -> int:
    """Ford-Fulkerson max-flow using BFS (Edmonds-Karp)."""
    from collections import deque
    n = len(graph)
    cap = [row[:] for row in graph]

    def bfs(s, t, parent):
        visited = {s}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in range(n):
                if v not in visited and cap[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == t:
                        return True
                    q.append(v)
        return False

    max_flow = 0
    while True:
        parent = [-1] * n
        if not bfs(source, sink, parent):
            break
        path_flow = float("inf")
        s = sink
        while s != source:
            u = parent[s]
            path_flow = min(path_flow, cap[u][s])
            s = parent[s]
        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            cap[u][v] -= path_flow
            cap[v][u] += path_flow
            v = parent[v]
    return max_flow
''',

    # ── String algorithms ─────────────────────────────────────────────────────
    '''\
def kmp_search(text: str, pattern: str) -> list[int]:
    """Knuth-Morris-Pratt substring search. Returns all start positions."""
    def build_lps(p: str) -> list[int]:
        lps = [0] * len(p)
        length = 0
        i = 1
        while i < len(p):
            if p[i] == p[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = build_lps(pattern)
    results: list[int] = []
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1; j += 1
        if j == len(pattern):
            results.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            j = lps[j - 1] if j else (i := i + 1) and 0
    return results
''',

    '''\
def z_algorithm(s: str) -> list[int]:
    """Z-array: z[i] = length of longest substring starting at s[i]
    that is also a prefix of s.  z[0] is defined as len(s).
    """
    n = len(s)
    z = [0] * n
    z[0] = n
    l = r = 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z
''',

    # ── Math / number theory ─────────────────────────────────────────────────
    '''\
def sieve_of_eratosthenes(limit: int) -> list[int]:
    """Return all primes up to limit using the Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i : limit + 1 : i] = bytearray(len(range(i * i, limit + 1, i)))
    return [i for i, p in enumerate(is_prime) if p]
''',

    '''\
def gcd(a: int, b: int) -> int:
    """Greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    """Least common multiple."""
    return a // gcd(a, b) * b

def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm: returns (gcd, x, y) such that a*x + b*y = gcd."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y
''',

    '''\
def fast_power(base: int, exp: int, mod: int | None = None) -> int:
    """Fast modular exponentiation using repeated squaring — O(log exp)."""
    result = 1
    base = base % mod if mod else base
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod if mod else result * base
        exp >>= 1
        base = (base * base) % mod if mod else base * base
    return result
''',

    '''\
def fibonacci(n: int) -> int:
    """n-th Fibonacci number using matrix exponentiation — O(log n)."""
    if n <= 0:
        return 0
    if n == 1:
        return 1

    def mat_mul(a, b):
        return [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0],
             a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0],
             a[1][0] * b[0][1] + a[1][1] * b[1][1]],
        ]

    def mat_pow(m, p):
        result = [[1, 0], [0, 1]]  # identity
        while p:
            if p & 1:
                result = mat_mul(result, m)
            m = mat_mul(m, m)
            p >>= 1
        return result

    m = [[1, 1], [1, 0]]
    return mat_pow(m, n)[0][1]
''',

    '''\
def is_prime(n: int) -> bool:
    """Miller-Rabin primality test (deterministic for n < 3.3 * 10^24)."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n in small_primes:
        return True
    if any(n % p == 0 for p in small_primes):
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True
''',

    # ── Geometry ─────────────────────────────────────────────────────────────
    '''\
def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain convex hull algorithm — O(n log n)."""
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]
''',

    # ── Miscellaneous ─────────────────────────────────────────────────────────
    '''\
def lru_cache_impl(capacity: int):
    """LRU cache factory returning get/put callables."""
    from collections import OrderedDict
    cache: OrderedDict = OrderedDict()

    def get(key):
        if key not in cache:
            return -1
        cache.move_to_end(key)
        return cache[key]

    def put(key, value):
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        if len(cache) > capacity:
            cache.popitem(last=False)

    return get, put
''',

    '''\
def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    """Return indices (i, j) where nums[i] + nums[j] == target, or None."""
    seen: dict[int, int] = {}
    for i, x in enumerate(nums):
        complement = target - x
        if complement in seen:
            return seen[complement], i
        seen[x] = i
    return None
''',

    '''\
def power_set(items: list) -> list[list]:
    """Return all subsets of items (power set)."""
    result = [[]]
    for item in items:
        result += [subset + [item] for subset in result]
    return result
''',

    '''\
def flatten(nested) -> list:
    """Recursively flatten a nested list of arbitrary depth."""
    result: list = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
''',

    '''\
def memoize(func):
    """Decorator that caches function calls by argument tuple."""
    cache: dict = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.__wrapped__ = func
    return wrapper
''',

    '''\
def sliding_window_max(nums: list[int], k: int) -> list[int]:
    """Maximum of each sliding window of size k — O(n) with deque."""
    from collections import deque
    dq: deque = deque()  # indices; front = max
    result: list[int] = []
    for i, x in enumerate(nums):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
''',

    '''\
def run_length_encode(s: str) -> str:
    """Run-length encoding: "aaabbc" -> "3a2b1c"."""
    if not s:
        return ""
    result: list[str] = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(f"{count}{s[i - 1]}")
            count = 1
    result.append(f"{count}{s[-1]}")
    return "".join(result)
''',

    '''\
class SegmentTree:
    """Segment tree for range-sum queries and point updates — O(log n) each."""
    def __init__(self, nums: list[int]) -> None:
        self._n = len(nums)
        self._tree = [0] * (2 * self._n)
        for i, v in enumerate(nums):
            self._tree[self._n + i] = v
        for i in range(self._n - 1, 0, -1):
            self._tree[i] = self._tree[2 * i] + self._tree[2 * i + 1]

    def update(self, pos: int, val: int) -> None:
        pos += self._n
        self._tree[pos] = val
        while pos > 1:
            pos //= 2
            self._tree[pos] = self._tree[2 * pos] + self._tree[2 * pos + 1]

    def query(self, lo: int, hi: int) -> int:
        """Sum of nums[lo:hi] (exclusive hi)."""
        res = 0
        lo += self._n; hi += self._n
        while lo < hi:
            if lo & 1:
                res += self._tree[lo]; lo += 1
            if hi & 1:
                hi -= 1; res += self._tree[hi]
            lo >>= 1; hi >>= 1
        return res
''',
]


# ── Source 2: builtin usage patterns ─────────────────────────────────────────

def _generate_builtin_patterns() -> list[str]:
    """Generate usage examples for common Python builtin functions."""
    patterns: list[str] = []

    patterns.append('''\
# Built-in functions: map, filter, reduce, zip, enumerate
from functools import reduce

def demo_builtins():
    nums = [1, 2, 3, 4, 5]
    doubled     = list(map(lambda x: x * 2, nums))
    evens       = list(filter(lambda x: x % 2 == 0, nums))
    total       = reduce(lambda a, b: a + b, nums)
    indexed     = list(enumerate(nums, start=1))
    paired      = list(zip(nums, doubled))
    return doubled, evens, total, indexed, paired
''')

    patterns.append('''\
# sorted() with key and reverse
def sort_examples():
    words = ["banana", "apple", "cherry", "date"]
    by_length   = sorted(words, key=len)
    by_last_ch  = sorted(words, key=lambda w: w[-1])
    descending  = sorted(words, reverse=True)
    pairs       = [(3, "c"), (1, "a"), (2, "b")]
    by_second   = sorted(pairs, key=lambda p: p[1])
    return by_length, by_last_ch, descending, by_second
''')

    patterns.append('''\
# list / dict / set comprehensions
def comprehension_examples():
    squares      = [x ** 2 for x in range(10)]
    even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
    matrix       = [[i * j for j in range(1, 4)] for i in range(1, 4)]
    word_lengths = {word: len(word) for word in ["hello", "world", "python"]}
    unique_chars = {ch for ch in "abracadabra"}
    return squares, even_squares, matrix, word_lengths, unique_chars
''')

    patterns.append('''\
# str methods
def string_methods_demo(s: str) -> dict:
    return {
        "upper":      s.upper(),
        "lower":      s.lower(),
        "strip":      s.strip(),
        "split":      s.split(),
        "replace":    s.replace("a", "A"),
        "startswith": s.startswith("he"),
        "endswith":   s.endswith("ld"),
        "find":       s.find("l"),
        "count":      s.count("l"),
        "join":       ", ".join(s.split()),
        "format":     "hello {}".format(s),
        "isdigit":    s.isdigit(),
        "isalpha":    s.isalpha(),
    }
''')

    patterns.append('''\
# dict methods
def dict_operations():
    d = {"a": 1, "b": 2, "c": 3}
    keys    = list(d.keys())
    values  = list(d.values())
    items   = list(d.items())
    default = d.get("z", 0)
    d.setdefault("d", 4)
    merged  = {**d, "e": 5}
    popped  = d.pop("a", None)
    return keys, values, items, default, merged, popped
''')

    patterns.append('''\
# itertools usage
import itertools

def itertools_demo():
    # Combinations and permutations
    combos  = list(itertools.combinations([1, 2, 3], 2))
    perms   = list(itertools.permutations([1, 2, 3], 2))
    product = list(itertools.product([0, 1], repeat=3))
    # Infinite iterators (take first 5)
    count5  = list(itertools.islice(itertools.count(10, 2), 5))
    # Chain and cycle
    chained = list(itertools.chain([1, 2], [3, 4], [5]))
    return combos, perms, product, count5, chained
''')

    patterns.append('''\
# functools
import functools

@functools.lru_cache(maxsize=None)
def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

def partial_example():
    power_of_two = functools.partial(pow, 2)
    return [power_of_two(i) for i in range(10)]
''')

    patterns.append('''\
# collections: Counter, defaultdict, namedtuple, deque
from collections import Counter, defaultdict, namedtuple, deque

def collections_demo():
    counter     = Counter("abracadabra")
    most_common = counter.most_common(3)
    dd          = defaultdict(list)
    for word in ["apple", "banana", "apricot", "blueberry"]:
        dd[word[0]].append(word)
    Point  = namedtuple("Point", ["x", "y"])
    origin = Point(0, 0)
    dq     = deque([1, 2, 3], maxlen=5)
    dq.appendleft(0)
    dq.rotate(1)
    return most_common, dict(dd), origin, list(dq)
''')

    return patterns


# ── Source 4: Augmentation ────────────────────────────────────────────────────

_VAR_RENAME_MAP = [
    {"x": "value",    "n": "count",   "i": "index",  "j": "pos",    "k": "step"},
    {"x": "element",  "n": "length",  "i": "idx",    "j": "offset", "k": "stride"},
    {"result": "output", "arr": "data",   "lst": "items", "tmp": "temp"},
]


class _VarRenamer(ast.NodeTransformer):
    """AST transformer that renames local variables according to a mapping."""

    def __init__(self, rename_map: dict[str, str]) -> None:
        self._map = rename_map

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self._map:
            node.id = self._map[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg in self._map:
            node.arg = self._map[node.arg]
        return node


def _augment_code(code: str, seed: int) -> list[str]:
    """Generate 2 augmented variants of *code* via AST variable renaming."""
    variants: list[str] = []
    rng = random.Random(seed)
    rename_maps = rng.sample(_VAR_RENAME_MAP, min(2, len(_VAR_RENAME_MAP)))

    for rename_map in rename_maps:
        try:
            tree = ast.parse(code)
            transformed = _VarRenamer(rename_map).visit(tree)
            ast.fix_missing_locations(transformed)
            new_code = ast.unparse(transformed)
            if new_code != code and new_code.strip():
                variants.append(new_code)
        except Exception:
            pass  # skip if AST transform fails

    return variants


# ── Source 1: Python stdlib ───────────────────────────────────────────────────

def _collect_stdlib_sources() -> list[tuple[str, float]]:
    """Collect (source_code, trust_score) pairs from stdlib modules."""
    samples: list[tuple[str, float]] = []
    for mod_name in STDLIB_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        # Collect all functions and classes
        try:
            members = inspect.getmembers(mod, predicate=lambda x: (
                inspect.isfunction(x) or inspect.isclass(x)
            ))
        except Exception:
            continue
        for name, obj in members:
            if name.startswith("_"):
                continue
            try:
                src = inspect.getsource(obj)
                # Quality filter: non-trivial source
                if len(src.strip()) >= 50:
                    samples.append((src, 95.0))
            except (OSError, TypeError):
                continue
    return samples


# ── Record builder ────────────────────────────────────────────────────────────

def _build_record(
    code: str,
    tokenizer,
    source: str,
    trust_score: float,
    max_seq_len: int,
) -> list[dict]:
    """Tokenize *code* into ≤max_seq_len records with shifted labels."""
    code = code.strip()
    if not code:
        return []

    # Encode without BOS/EOS — we'll add them manually
    token_ids = tokenizer.encode(code, add_bos=False, add_eos=False)
    if not token_ids:
        return []

    # Prepend BOS, append EOS
    full_ids = [tokenizer.bos_id] + token_ids + [tokenizer.eos_id]

    records: list[dict] = []
    # Chunk into windows of max_seq_len + 1 so shifted labels align
    window = max_seq_len + 1
    for start in range(0, len(full_ids) - 1, max_seq_len):
        chunk = full_ids[start: start + window]
        if len(chunk) < 2:
            break
        input_ids = chunk[:-1]     # length ≤ max_seq_len
        labels    = chunk[1:]      # same length, shifted by 1

        # Pad to max_seq_len
        pad_len = max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.pad_id] * pad_len
            labels    = labels    + [-100]              * pad_len

        records.append({
            "input_ids":   input_ids,
            "labels":      labels,
            "trust_score": trust_score,
            "source":      source,
        })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Aether Omega training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output",           type=str, default="data/aether_train.jsonl")
    p.add_argument("--tokenizer",        type=str, default="omega_tokenizer.json")
    p.add_argument("--n-samples",        type=int, default=60_000)
    p.add_argument("--max-seq-len",      type=int, default=512)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--train-tokenizer",  action="store_true",
                   help="Train a new tokenizer from the corpus before generating data")
    p.add_argument("--tokenizer-vocab",  type=int, default=16_000)
    return p.parse_args()


def main() -> None:
    args = _parse()
    rng  = random.Random(args.seed)
    out  = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Aether Omega — Training Data Generator")
    print(f"  output    : {out}")
    print(f"  tokenizer : {args.tokenizer}")
    print(f"  n_samples : {args.n_samples:,}")
    print(f"  max_seq   : {args.max_seq_len}")
    print(f"  seed      : {args.seed}")
    print("=" * 60)

    # ── Collect raw text corpus ────────────────────────────────────────────────

    print("\n[1/4] Collecting Python stdlib sources …")
    stdlib_samples = _collect_stdlib_sources()
    print(f"      {len(stdlib_samples):,} stdlib functions/classes found")

    print("[2/4] Building builtin usage patterns …")
    builtin_texts = _generate_builtin_patterns()
    builtin_samples = [(t, 90.0) for t in builtin_texts]
    print(f"      {len(builtin_samples):,} builtin patterns")

    print("[3/4] Loading algorithmic patterns …")
    algo_samples = [(t, 92.0) for t in ALGO_PATTERNS]
    print(f"      {len(algo_samples):,} algorithm implementations")

    all_raw: list[tuple[str, float, str]] = (
        [(t, s, "stdlib")  for t, s in stdlib_samples] +
        [(t, s, "builtin") for t, s in builtin_samples] +
        [(t, s, "algo")    for t, s in algo_samples]
    )
    print(f"\n      Total raw corpus: {len(all_raw):,} code texts")

    # ── Optionally train tokenizer ─────────────────────────────────────────────

    if args.train_tokenizer:
        print("\n[tokenizer] Training tokenizer from corpus …")
        from tokenizer import OmegaTokenizer
        tok = OmegaTokenizer()
        corpus_texts = [t for t, _, _ in all_raw]
        tok.train(corpus_texts, vocab_size=args.tokenizer_vocab)
        tok.save(args.tokenizer)
    else:
        from tokenizer import OmegaTokenizer
        if not Path(args.tokenizer).exists():
            print(f"\n[error] Tokenizer not found: {args.tokenizer}")
            print("  Run with --train-tokenizer to train one first, or:")
            print("  python tokenizer.py --train <corpus> --output omega_tokenizer.json")
            sys.exit(1)
        tok = OmegaTokenizer.from_file(args.tokenizer)
        print(f"\n[tokenizer] Loaded: vocab_size={tok.vocab_size}")

    if tok.vocab_size != 16_000:
        print(f"[warning] Tokenizer vocab_size={tok.vocab_size} "
              f"(expected 16000 to match OmegaConfig.vocab_size)")

    # ── Generate augmented samples ─────────────────────────────────────────────

    print("\n[4/4] Generating augmented samples …")
    augmented: list[tuple[str, float, str]] = []
    for code, trust, src in all_raw:
        variants = _augment_code(code, seed=rng.randint(0, 2 ** 31))
        for v in variants:
            augmented.append((v, round(trust * 0.85, 1), "augmented"))

    all_samples = all_raw + augmented
    rng.shuffle(all_samples)
    print(f"      {len(augmented):,} augmented variants")
    print(f"      Total samples (before tokenization): {len(all_samples):,}")

    # ── Tokenize and write ─────────────────────────────────────────────────────

    written = 0
    source_counts: dict[str, int] = {}
    total_tokens = 0

    with out.open("w", encoding="utf-8") as fh:
        for code, trust, src in all_samples:
            if written >= args.n_samples:
                break
            records = _build_record(code, tok, src, trust, args.max_seq_len)
            for rec in records:
                if written >= args.n_samples:
                    break
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                written += 1
                source_counts[src] = source_counts.get(src, 0) + 1
                total_tokens += len([x for x in rec["input_ids"] if x != tok.pad_id])

    print(f"\n{'=' * 60}")
    print(f"  Done: {written:,} samples written → {out}")
    print(f"  File size: {out.stat().st_size / 1e6:.1f} MB")
    print(f"\n  Sources:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / max(written, 1)
        print(f"    {src:<14}  {cnt:>8,}  ({pct:.1f}%)")
    avg_len = total_tokens / max(written, 1)
    print(f"\n  Avg tokens/sample : {avg_len:.0f}")
    print(f"  Total tokens      : {total_tokens:,}")
    # Vocabulary coverage
    seen_ids: set[int] = set()
    with out.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            seen_ids.update(x for x in rec["input_ids"] if x >= 4)
    cov = 100 * len(seen_ids) / max(tok.vocab_size - 4, 1)
    print(f"  Vocab coverage    : {len(seen_ids):,} / {tok.vocab_size - 4:,} ({cov:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
