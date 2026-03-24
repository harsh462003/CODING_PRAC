# 10 Easy DSA Solved Problems

## 1) Two Sum
**Problem:** Given an array `nums` and target `target`, return indices of two numbers that add up to target.

**Solution (Python):**
```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []
```
**Explanation:** Store visited numbers with indices in a hash map. For each value `x`, check if `target-x` exists.

**Complexity:** `O(n)` time, `O(n)` space.

---

## 2) Best Time to Buy and Sell Stock
**Problem:** Maximize profit from one buy and one sell.

```python
def max_profit(prices):
    min_price = float('inf')
    ans = 0
    for p in prices:
        min_price = min(min_price, p)
        ans = max(ans, p - min_price)
    return ans
```
**Explanation:** Track minimum price so far and best profit at each day.

**Complexity:** `O(n)` time, `O(1)` space.

---

## 3) Valid Parentheses
**Problem:** Check if string of brackets is valid.

```python
def is_valid(s):
    mp = {')': '(', ']': '[', '}': '{'}
    st = []
    for ch in s:
        if ch in '([{':
            st.append(ch)
        else:
            if not st or st[-1] != mp[ch]:
                return False
            st.pop()
    return len(st) == 0
```
**Explanation:** Use stack; each closing bracket must match the latest opening.

**Complexity:** `O(n)` time, `O(n)` space.

---

## 4) Merge Two Sorted Lists
**Problem:** Merge two sorted linked lists.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_two_lists(l1, l2):
    dummy = ListNode()
    cur = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            cur.next, l1 = l1, l1.next
        else:
            cur.next, l2 = l2, l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next
```
**Explanation:** Two pointers compare current nodes and append smaller one.

**Complexity:** `O(n+m)` time, `O(1)` extra space.

---

## 5) Binary Search
**Problem:** Find target index in sorted array.

```python
def binary_search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            return m
        if nums[m] < target:
            l = m + 1
        else:
            r = m - 1
    return -1
```
**Explanation:** Repeatedly halve search space based on middle comparison.

**Complexity:** `O(log n)` time, `O(1)` space.

---

## 6) Maximum Subarray (Kadane)
**Problem:** Find contiguous subarray with max sum.

```python
def max_subarray(nums):
    cur = best = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
```
**Explanation:** At each index, decide start new subarray or extend previous best ending here.

**Complexity:** `O(n)` time, `O(1)` space.

---

## 7) Move Zeroes
**Problem:** Move all zeroes to end while keeping order of non-zero elements.

```python
def move_zeroes(nums):
    insert = 0
    for x in nums:
        if x != 0:
            nums[insert] = x
            insert += 1
    while insert < len(nums):
        nums[insert] = 0
        insert += 1
```
**Explanation:** Compact non-zeroes first, then fill remaining positions with zero.

**Complexity:** `O(n)` time, `O(1)` space.

---

## 8) Missing Number
**Problem:** Array contains `n` distinct numbers from `[0..n]`, find missing one.

```python
def missing_number(nums):
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)
```
**Explanation:** Expected sum minus actual sum gives missing value.

**Complexity:** `O(n)` time, `O(1)` space.

---

## 9) Invert Binary Tree
**Problem:** Swap left and right children recursively.

```python
def invert_tree(root):
    if not root:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```
**Explanation:** Post-order style recursive swapping.

**Complexity:** `O(n)` time, `O(h)` recursion stack.

---

## 10) Palindrome Number
**Problem:** Check if integer reads same backward.

```python
def is_palindrome(x):
    if x < 0:
        return False
    return str(x) == str(x)[::-1]
```
**Explanation:** Negative numbers fail; compare string with reverse.

**Complexity:** `O(d)` time, `O(d)` space where `d` is number of digits.
