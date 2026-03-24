# 10 Hard DSA Solved Problems

## 1) Median of Two Sorted Arrays
```python
def find_median_sorted_arrays(a, b):
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2
        j = (m + n + 1) // 2 - i

        aL = a[i-1] if i > 0 else float('-inf')
        aR = a[i] if i < m else float('inf')
        bL = b[j-1] if j > 0 else float('-inf')
        bR = b[j] if j < n else float('inf')

        if aL <= bR and bL <= aR:
            if (m + n) % 2:
                return max(aL, bL)
            return (max(aL, bL) + min(aR, bR)) / 2
        elif aL > bR:
            hi = i - 1
        else:
            lo = i + 1
```
**Explanation:** Binary-search partition of smaller array so left-half elements are <= right-half.
**Complexity:** `O(log(min(m,n)))` time.

---

## 2) Trapping Rain Water
```python
def trap(height):
    l, r = 0, len(height) - 1
    left_max = right_max = 0
    ans = 0
    while l < r:
        if height[l] < height[r]:
            left_max = max(left_max, height[l])
            ans += left_max - height[l]
            l += 1
        else:
            right_max = max(right_max, height[r])
            ans += right_max - height[r]
            r -= 1
    return ans
```
**Explanation:** Two pointers with running max walls; trapped water depends on smaller side.
**Complexity:** `O(n)` time, `O(1)` space.

---

## 3) Merge k Sorted Lists
```python
import heapq

def merge_k_lists(lists):
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    dummy = ListNode(0)
    cur = dummy
    while heap:
        _, i, node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
```
**Explanation:** Min-heap always extracts smallest head among k lists.
**Complexity:** `O(N log k)` time.

---

## 4) Serialize and Deserialize Binary Tree
```python
from collections import deque

def serialize(root):
    if not root:
        return ''
    q, out = deque([root]), []
    while q:
        node = q.popleft()
        if node:
            out.append(str(node.val))
            q.append(node.left)
            q.append(node.right)
        else:
            out.append('#')
    return ','.join(out)


def deserialize(data):
    if not data:
        return None
    vals = data.split(',')
    root = TreeNode(int(vals[0]))
    q = deque([root])
    i = 1
    while q:
        node = q.popleft()
        if vals[i] != '#':
            node.left = TreeNode(int(vals[i]))
            q.append(node.left)
        i += 1
        if vals[i] != '#':
            node.right = TreeNode(int(vals[i]))
            q.append(node.right)
        i += 1
    return root
```
**Explanation:** Level-order encoding with null markers preserves tree shape.
**Complexity:** `O(n)` for both.

---

## 5) Largest Rectangle in Histogram
```python
def largest_rectangle_area(heights):
    st = []
    ans = 0
    heights.append(0)
    for i, h in enumerate(heights):
        while st and heights[st[-1]] > h:
            H = heights[st.pop()]
            L = st[-1] if st else -1
            ans = max(ans, H * (i - L - 1))
        st.append(i)
    heights.pop()
    return ans
```
**Explanation:** Monotonic stack finds nearest smaller left/right boundaries.
**Complexity:** `O(n)` time.

---

## 6) Word Ladder
```python
from collections import deque, defaultdict

def ladder_length(begin, end, word_list):
    words = set(word_list)
    if end not in words:
        return 0
    pat = defaultdict(list)
    L = len(begin)
    for w in words:
        for i in range(L):
            pat[w[:i] + '*' + w[i+1:]].append(w)

    q = deque([(begin, 1)])
    vis = {begin}
    while q:
        w, d = q.popleft()
        if w == end:
            return d
        for i in range(L):
            p = w[:i] + '*' + w[i+1:]
            for nxt in pat[p]:
                if nxt not in vis:
                    vis.add(nxt)
                    q.append((nxt, d + 1))
    return 0
```
**Explanation:** BFS on implicit graph where words differ by one character.
**Complexity:** roughly `O(N * L^2)`.

---

## 7) N-Queens
```python
def solve_n_queens(n):
    cols, d1, d2 = set(), set(), set()
    board = [['.'] * n for _ in range(n)]
    ans = []

    def backtrack(r):
        if r == n:
            ans.append([''.join(row) for row in board])
            return
        for c in range(n):
            if c in cols or (r - c) in d1 or (r + c) in d2:
                continue
            cols.add(c); d1.add(r - c); d2.add(r + c)
            board[r][c] = 'Q'
            backtrack(r + 1)
            board[r][c] = '.'
            cols.remove(c); d1.remove(r - c); d2.remove(r + c)

    backtrack(0)
    return ans
```
**Explanation:** Backtracking with constraints for columns and diagonals.
**Complexity:** Exponential (`O(n!)` upper bound).

---

## 8) Sliding Window Maximum
```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()
    ans = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            ans.append(nums[dq[0]])
    return ans
```
**Explanation:** Deque stores indices in decreasing values; front is max of current window.
**Complexity:** `O(n)` time.

---

## 9) Minimum Window Substring
```python
from collections import Counter

def min_window(s, t):
    need = Counter(t)
    missing = len(t)
    l = start = end = 0
    for r, ch in enumerate(s, 1):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        if missing == 0:
            while l < r and need[s[l]] < 0:
                need[s[l]] += 1
                l += 1
            if end == 0 or r - l < end - start:
                start, end = l, r
            need[s[l]] += 1
            missing += 1
            l += 1
    return s[start:end]
```
**Explanation:** Expand window to satisfy all chars, then shrink to minimal valid window.
**Complexity:** `O(n)` time.

---

## 10) Edit Distance
```python
def min_distance(w1, w2):
    m, n = len(w1), len(w2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if w1[i-1] == w2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```
**Explanation:** DP over prefixes with insert/delete/replace transitions.
**Complexity:** `O(mn)` time, `O(mn)` space.
