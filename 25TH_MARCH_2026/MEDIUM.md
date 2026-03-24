# 10 Medium DSA Solved Problems

## 1) 3Sum
```python
def three_sum(nums):
    nums.sort()
    ans = []
    n = len(nums)
    for i in range(n):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                ans.append([nums[i], nums[l], nums[r]])
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l-1]:
                    l += 1
                while l < r and nums[r] == nums[r+1]:
                    r -= 1
            elif s < 0:
                l += 1
            else:
                r -= 1
    return ans
```
**Explanation:** Sort + fix one element, then two-pointer search for remaining pair.
**Complexity:** `O(n^2)` time, `O(1)` extra.

---

## 2) Longest Substring Without Repeating Characters
```python
def length_of_longest_substring(s):
    seen = {}
    l = ans = 0
    for r, ch in enumerate(s):
        if ch in seen and seen[ch] >= l:
            l = seen[ch] + 1
        seen[ch] = r
        ans = max(ans, r - l + 1)
    return ans
```
**Explanation:** Sliding window with last seen index to avoid duplicates.
**Complexity:** `O(n)` time, `O(k)` space.

---

## 3) Product of Array Except Self
```python
def product_except_self(nums):
    n = len(nums)
    ans = [1] * n
    pref = 1
    for i in range(n):
        ans[i] = pref
        pref *= nums[i]
    suff = 1
    for i in range(n-1, -1, -1):
        ans[i] *= suff
        suff *= nums[i]
    return ans
```
**Explanation:** Build prefix product then multiply suffix product in reverse.
**Complexity:** `O(n)` time, `O(1)` extra (excluding output).

---

## 4) Group Anagrams
```python
from collections import defaultdict

def group_anagrams(strs):
    mp = defaultdict(list)
    for w in strs:
        key = ''.join(sorted(w))
        mp[key].append(w)
    return list(mp.values())
```
**Explanation:** Sorted word is canonical key for its anagram group.
**Complexity:** `O(n * m log m)` time.

---

## 5) Rotate Image
```python
def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
```
**Explanation:** Transpose matrix, then reverse each row for 90° clockwise rotation.
**Complexity:** `O(n^2)` time, `O(1)` space.

---

## 6) Kth Largest Element in an Array
```python
import heapq

def find_kth_largest(nums, k):
    return heapq.nlargest(k, nums)[-1]
```
**Explanation:** Use heap utilities to obtain k largest elements.
**Complexity:** `O(n log k)` time.

---

## 7) Coin Change
```python
def coin_change(coins, amount):
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return -1 if dp[amount] == INF else dp[amount]
```
**Explanation:** Bottom-up DP where `dp[a]` is min coins to make amount `a`.
**Complexity:** `O(amount * len(coins))` time.

---

## 8) Number of Islands
```python
def num_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if r < 0 or c < 0 or r >= rows or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'
        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    return count
```
**Explanation:** DFS flood-fill each unvisited land cell.
**Complexity:** `O(mn)` time.

---

## 9) Validate Binary Search Tree
```python
def is_valid_bst(root):
    def dfs(node, low, high):
        if not node:
            return True
        if not (low < node.val < high):
            return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)
    return dfs(root, float('-inf'), float('inf'))
```
**Explanation:** Every node value must be within allowed range inherited from ancestors.
**Complexity:** `O(n)` time, `O(h)` space.

---

## 10) Top K Frequent Elements
```python
from collections import Counter

def top_k_frequent(nums, k):
    cnt = Counter(nums)
    return [x for x, _ in cnt.most_common(k)]
```
**Explanation:** Frequency map and retrieve top `k` by count.
**Complexity:** `O(n log n)` worst-case.
