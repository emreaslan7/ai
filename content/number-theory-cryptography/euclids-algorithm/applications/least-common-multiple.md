# Least Common Multiple

&emsp; The least common multiple ($lcm$) is a fundamental concept in number theory, with applications ranging from simple arithmetic to advanced algorithms.

In essence, the $lcm$ of two integers is the smallest positive integer that is divisible by both numbers.

This concept is crucial in various mathematical operations, especially when dealing with fractions and rational numbers.

## Definition and Properties

&emsp; Formally, the least common multiple of two non-zero integers, $a$ and $b$, denoted as $lcm(a, b)$, is the smallest positive integer that is divisible by both $a$ and $b$. It is undefined if either $a$ or $b$ is zero.

### Key properties of the $lcm$ include:

- <b>Commutativity:</b>&emsp; $lcm(a, b) = lcm(b, a)$

- <b>Associativity:</b>&emsp; $lcm(a, lcm(b, c)) = lcm(lcm(a, b), c)$

- <b>Distributivity:</b>&emsp; $lcm(a, bc) = lcm(lcm(a, b), c)$

- <b>Relationship with Greatest Common Divisor ($gcd$):</b>&emsp; $lcm(a, b) \cdot gcd(a, b) = |ab|$

## Applications of the $lcm$

The $lcm$ finds numerous applications in various mathematical and computational contexts:

- <b>Fractions:</b>&emsp; When adding or subtracting fractions, the lcm of the denominators is used to find a common denominator.

- <b>Modular Arithmetic:</b>&emsp; The $lcm$ is used to determine the least common period of periodic functions in modular arithmetic.

- <b>Scheduling Problems:</b>&emsp; In scheduling problems, the $lcm$ is used to find the earliest time at which two or more events can occur simultaneously.

- <b>Number Theory:</b>&emsp; The $lcm$ is a fundamental concept in number theory, used in various theorems and algorithms.

## Algorithms for Computing the $lcm$

Several algorithms can be used to compute the $lcm$ of two integers:

### Brute Force Method

The simplest approach is to iterate through all integers from $1$ to the product of $a$ and $b$, checking if each integer is divisible by both $a$ and $b$. The first such integer is the $lcm$.

```python
def lcm_naive(a, b):

  max_num = max(a, b)

  while True:
    if max_num % a == 0 and max_num % b == 0:
      return max_num
    max_num += 1
```

### Using the $gcd$

&emsp;A more efficient method is to use the relationship between the $lcm$ and the $gcd$:

The $lcm$ and $gcd$ of two numbers are closely related. The equation $$lcm(a,b) \cdot gcd(a,b) = a \cdot b$$ states that the product of the $lcm$ and $gcd$ of two numbers is equal to the product of the two numbers.

When we multiply the $gcd$ and $lcm$, we are essentially "canceling out" the common factors and ending up with the product of the original numbers.

```python
def gcd(a, b):
    #Calculates the GCD of two numbers using the Euclidean algorithm.

    while b != 0:
        a, b = b, a % b

return a
```

```python
def lcm_efficient(a, b):
    #Calculates the LCM of two numbers using the GCD.

return (a * b) // gcd(a, b)
```

#### Why is this more efficient?

- <b>Naive approach:</b>&emsp; Simple to understand but can be very slow for large numbers, especially if the $lcm$ is much larger than the inputs.

- <b>Euclidean algorithm approach:</b>&emsp; More complex but significantly faster, especially for larger numbers. It leverages a well-established algorithm for calculating the $gcd$ and a direct formula to compute the $lcm$.

In conclusion, while the naive approach is straightforward, the Euclidean algorithm-based approach is generally preferred for its efficiency and elegance. The relationship between $lcm$ and $gcd$ provides a powerful tool for calculating the $lcm$ in a computationally efficient manner.

## Conclusion

&emsp; The least common multiple is a fundamental concept in number theory with widespread applications. Understanding its properties and algorithms for computing it is essential for various mathematical operations and problem-solving tasks.
