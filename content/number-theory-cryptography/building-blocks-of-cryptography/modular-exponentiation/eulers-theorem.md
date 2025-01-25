# Euler's Totient Theorem

&emsp; At its core lies **Euler’s Totient Function**, denoted $\phi(n)$, which counts the integers up to $n$ that are coprime with $n$. This seemingly simple concept has profound implications, ranging from pure mathematics to cryptography.

## Key Concepts in Euler's Totient Theorem

### 1. **Definition of $\phi(n)$:**

The totient function $\phi(n)$ is defined as:

$$
\phi(n) = \text{Number of integers } k \text{ such that } 1 \leq k \leq n \text{ and } \text{gcd}(k, n) = 1.
$$

##### Example:

For $n = 12$:

$$
{1, \cancel{2}, \cancel{3}, \cancel{4}, 5, \cancel{6}, 7, \cancel{8}, \cancel{9}, \cancel{10}, 11}. \\ \text{Coprime integers: }{[1,5,7,11]}
$$

Thus, $\phi(12) = 4$.

#### What happens with prime numbers?

Prime numbers has only one prime factor: itself!

##### Example

For $n = 5$:

$$
{1, 2, 3, 4, \cancel 5}. \\ \text{Coprime integers: }{[1,2,3,4]}
$$

Thus, $\phi(5) = 4$.

This is generally true of all prime numbers:

$$\phi(p) = p − 1  \\ \text{where p is a prime number} $$

We have our first timesaver.

---

### 2. **Multiplicative Property:**

If $m$ and $n$ are coprime ($\text{gcd}(m, n) = 1$), then:

$$
\phi(mn) = \phi(m) \cdot \phi(n).
$$

#### Example:

For $m = 5, n = 6$:

$$
\phi(5) = 4, \\ \phi(6) = 2, \\ \phi(30) = \phi(5) \ \cdot \phi(6) = 4 \cdot 2 = 8.
$$

---

### 3. **Prime Factorization Formula:**

For $n = p_1^{\alpha_1} p_2^{\alpha_2} \cdots p_k^{\alpha_k}$:

$$
\phi(n) = n \cdot \prod_{i=1}^k \left(1 - \frac{1}{p_i}\right)
$$

#### Example:

For $n = 60 = 2^2 \cdot 3 \cdot 5$:

$$
\phi(60) = 60 \cdot \left(1 - \frac{1}{2}\right) \cdot \left(1 - \frac{1}{3}\right) \cdot \left(1 - \frac{1}{5}\right),
$$

$$
\phi(60) = 60 \cdot \frac{1}{2} \cdot \frac{2}{3} \cdot \frac{4}{5} = 16.
$$

---

## Applications of Euler's Totient Theorem

### 1. **Primality Testing:**

For a prime $p$:

$$
\phi(p) = p - 1.
$$

This property aids in testing whether a number is prime.

---

### 2. **RSA Encryption:**

RSA relies on the totient function to generate keys:

- Let $n = p \cdot q$, where $p, q$ are primes.
  $$
  \phi(n) = (p-1)(q-1).
  $$

**Example:**
For $p = 11, q = 13$:

$$
n = 143, \\ \phi(143) = 10 \cdot 12 = 120.
$$

Public and private keys are derived using $\phi(n)$, ensuring secure encryption.

### 3. **Applications in Modular Arithmetic and Prime Numbers:**

#### a) **Efficient Exponentiation:**

Using the property $a^{\phi(n)} \equiv 1 \pmod{n}$ for $a$ coprime to $n$, large powers modulo $n$ can be computed efficiently.

**Example:**
Calculate $7^{100} \pmod{12}$:

1. Since $\phi(12) = 4$, $7^4 \equiv 1 \pmod{12}$.
2. Break $100$ into $4 \cdot 25$: $7^{100} = (7^4)^{25} \equiv 1^{25} \equiv 1 \pmod{12}$.

---

#### b) **Properties of Primes in Modular Arithmetic:**

Prime numbers simplify modular inverses, as every nonzero element in $\mathbb{Z}_p$ is coprime to $p$.

##### Example: Modular Inverses:

Find the modular inverse of $3 \pmod{7}$:

1. $\phi(7) = 6$, so $3^6 \equiv 1 \pmod{7}$.
2. Use powers to find $3 \cdot x \equiv 1 \pmod{7}$:
   $$3 \cdot 5 \equiv 15 \equiv 1 \pmod{7}.$$
   Thus, $5$ is the modular inverse of $3$ under mod $7$.

---

&emsp; Euler's Totient Theorem stands as a testament to the interplay between theory and application, bridging the gap between abstract mathematics and real-world cryptography. Its elegance and utility continue to inspire mathematicians and engineers alike.

<br/>
<br/>
