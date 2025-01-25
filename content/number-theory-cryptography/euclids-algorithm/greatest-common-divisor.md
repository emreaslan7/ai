# Greatest Common Divisor

&emsp; The greatest common divisor (GCD) of two integers, $A$ and $B$, is the largest positive integer that divides both A and B without leaving a remainder. It's a fundamental concept in number theory with applications in various fields, including cryptography and computer science.

## Key Properties of the GCD

- <b>Divisibility:</b> &emsp; Any common divisor of $A$ and $B$ must also divide their GCD.
- <b>Uniqueness:</b> &emsp; For any given $A$ and $B$, there exists a unique GCD.
- <b>Non-Trivial Case:</b> &emsp; If $A$ and $B$ are not both zero, their GCD is always positive.

## The Naive Approach: Inefficient Brute Force

&emsp; A straightforward method to compute the GCD is to iteratively check all possible divisors from the minimum of $A$ and $B$ down to $1$.

However, this approach becomes extremely inefficient for large numbers, as the number of iterations grows linearly with the minimum value.

You can try this python code on your computer. First try small numbers you see solution in miliseconds but if you try large numbers you can hear the sound of the computer's fans 😁

```python
def gcd(a, b):
  assert a >= 0 and b >= 0 and a + b > 0

  if a == 0 or b == 0:
    return max(a, b)

  for d in range(min(a, b), 0, -1):
    if a % d == 0 and b % d == 0:
      return d

  return 1

print(gcd(0, 1))
print(gcd(24, 16))
# The following call would take too long
#print(gcd(790933790547, 1849639579327))
```

## Motivation for an Efficient Algorithm

&emsp; The need for an efficient GCD algorithm arises from its applications in computationally intensive tasks. For example, in cryptography, the GCD is used in algorithms like the RSA encryption scheme. A slow GCD algorithm can significantly impact the performance of these cryptographic systems.

## The Euclidean Algorithm: A More Efficient Approach

&emsp; The Euclidean algorithm is a classical method for efficiently computing the greatest common divisor (GCD) of two numbers. It is based on the principle that the GCD of two numbers also divides their difference. This algorithm is not only simple but also significantly faster than the naive approach of checking all possible divisors.

In the next lesson, we will delve into the Euclidean algorithm in detail. We will explore its mathematical foundation, step-by-step procedure, and various optimizations that can be applied to further enhance its performance. Additionally, we will implement the algorithm in code and discuss its applications in different computational tasks, including cryptography and number theory.
