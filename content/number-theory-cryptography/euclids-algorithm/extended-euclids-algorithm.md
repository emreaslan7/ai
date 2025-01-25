# Extended Euclid's Algorithm

## Introduction

&emsp; The Extended Euclidean Algorithm is a fundamental algorithm in number theory that not only computes the greatest common divisor (GCD) of two integers but also provides coefficients that express the GCD as a linear combination of these integers. $$ gcd(a,b) = a \cdot x + b \cdot y$$

This additional information is invaluable in various mathematical and cryptographic applications.

## The Standard Euclidean Algorithm

&emsp; Before delving into the extended version, let's briefly review the standard Euclidean Algorithm. It's a method for finding the GCD of two non-negative integers.

The algorithm repeatedly applies the division algorithm to obtain a sequence of remainders until a remainder of $0$ is reached. The last non-zero remainder is the GCD.

### Example: &emsp; $gcd(888, 54)$

Let's apply the Euclidean algorithm to find the GCD of $888$ and $54$:

- Step 1:&emsp; $888 = 54 \cdot (16) + 24$

- Step 2:&emsp; $ 54 = 24 \cdot (2) + 6$

- Step 3:&emsp; $ 24 = 2 \cdot (12) + 0$

Since the remainder is $0$, the GCD is the last non-zero remainder, which is $6$.

$$gcd(888, 54) = 6$$

In summary, the Euclidean algorithm is a simple yet efficient method for finding the GCD of two numbers.

It involves repeated division and remainder calculations until a remainder of $0$ is reached. The last non-zero remainder is the GCD.

## Extending the Algorithm

&emsp; The Extended Euclidean Algorithm builds upon the standard algorithm by introducing two auxiliary sequences of integers. These sequences, often denoted as $x$ and $y$, are calculated at each step of the algorithm and are used to express the current remainder as a linear combination of the original two numbers.

### Formal Description:

Given two non-negative integers $a$ and $b$, the Extended Euclidean Algorithm calculates integers $g$, $x$, and $y$ such that:

$$g = gcd(a, b)$$
$$g = ax + by$$

Our main goal is to see which $x$ and $y$ value multiplication produces the number pair whose gcd we know. We are actually putting Euclid's operations in reverse order.

### Example: &emsp; $gcd(888, 54) = 888x + 54y$

#### Given:

$a = 888$ &emsp; $b = 54$ &emsp; $gcd(888, 54) = 6$ &emsp; (as calculated using the standard Euclidean algorithm)

#### Goal:

Find integers $x$ and $y$ such that &emsp; $6 = 888x + 54y$

#### Steps:

- Step 1:&emsp; $6= 54 - 24 \cdot 2$ &emsp;So, &emsp;$6= 54 + 24 \cdot (-2)$

- Step 2:&emsp; $6 = 54 + (888 - 54 \cdot (16) ) \cdot (-2)$

- Step 3:&emsp; $6 = 54 + 888 \cdot (-2) + 54 \cdot (-32)$&emsp; So,&emsp; $6= 54 \cdot (-33) + 888 \cdot (-2)$

Rearranging to match the desired form: $$6 = 888 \cdot (-2) + 54 \cdot 33$$

##### Final Solution:

Therefore, for the equation $6 = 888x + 54y$, the values of $x$ and $y$ are:

$$x = -2$$
$$y = 33$$

So, $gcd(888, 54)$ can be expressed as:

$$6 = 888 \cdot (-2) + 54 \cdot 33$$

## The Algorithm

```python
def extended_gcd(a, b):
  assert a >= b and b >= 0 and a + b > 0

  if b == 0:
    d, x, y = a, 1, 0
  else:
    (d, p, q) = extended_gcd(b, a % b)
    x = q
    y = p - q * (a // b)

  assert a % d == 0 and b % d == 0
  assert d == a * x + b * y
  return (d, x, y)
```

### Explanation

- Function Definition:

  <b>extended_gcd(a, b):</b>&emsp; This defines a function that takes two non-negative integers, $a$ and $b$, as input and returns a tuple containing the greatest common divisor (GCD) d, and the coefficients $x$ and $y$ such that $d = ax + by$.

- Preconditions:

  <b>assert a >= b and b >= 0 and a + b > 0:</b>&emsp; Ensures that the input numbers are valid for the algorithm. a must be greater than or equal to b, both must be non-negative, and their sum must be greater than $0$.

- Base Case:

  <b>if b == 0: d, x, y = a, 1, 0:</b>&emsp; If b is 0, the GCD is $a$, and the coefficients are $1$ and $0$, respectively. This is the base case of the recursion.

- Recursive Case:

  <b>(d, p, q) = extended_gcd(b, a % b):</b>&emsp; Recursively calls the function with b and the remainder of $a$ divided by $b$. This is the core of the Euclidean algorithm.

  <b>x = q:</b>&emsp; The coefficient $x$ is assigned the value of $q$ from the recursive call.

  <b>y = p - q \* (a // b):</b>&emsp; The coefficient $y$ is calculated using the values of $p$, $q$, and the integer division of $a$ by $b$. This step is crucial for expressing the GCD as a linear combination of $a$ and $b$.

- Postconditions:

  <b>assert a % d == 0 and b % d == 0:</b>&emsp; Verifies that the calculated GCD $d$ divides both $a$ and $b$.

  <b>assert d == a _ x + b _ y:</b>&emsp; Ensures that the calculated GCD $d$ can be expressed as a linear combination of $a$ and $b$ using the coefficients $x$ and $y$.

- Return Value:

  <b>return (d, x, y):</b>&emsp; Returns a tuple containing the GCD $d$ and the coefficients $x$ and $y$.

## Applications

The Extended Euclidean Algorithm has numerous applications, including:

- Modular Arithmetic:

  - Finding modular inverses:&emsp; Given two relatively prime integers $a$ and $n$, the algorithm can be used to find an integer $x$ such that $a \cdot x ≡ 1 (mod n)$ . This is crucial in cryptography for operations like decryption.

  - Solving linear Diophantine equations:&emsp; Equations of the form $a \cdot x + b \cdot y = c$ can be solved efficiently using the algorithm.

- Cryptography:

  - RSA algorithm:&emsp; The Extended Euclidean Algorithm is used to compute the private key in the RSA cryptosystem.

  - Diffie-Hellman key exchange:&emsp; It is used to compute the shared secret key.

- Number theory:

  - Continued fractions:&emsp; The algorithm can be used to find the continued fraction representation of a rational number.

## Conclusion

&emsp; The Extended Euclidean Algorithm is a fundamental algorithm with far-reaching implications in number theory and cryptography.

Its ability to compute the GCD and provide coefficients for a linear combination makes it an essential tool for solving various mathematical problems.

By understanding the principles behind this algorithm, one can gain a deeper appreciation for its applications in modern cryptography and computer science.
