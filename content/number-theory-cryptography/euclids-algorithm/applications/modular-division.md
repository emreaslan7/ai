# Modular Division

Modular division is a mathematical operation that involves finding the remainder when a number is divided by another number called the modulus.

## Multiple Inverse

In modular arithmetic, a modular inverse of a number a modulo $m$ is $a$ number $b$ such that:

$$a \cdot b ≡ 1 (mod m)$$

This means that when you multiply $a$ and $b$ and then divide by $m$, the remainder is $1$. Think of it like finding a number that "undoes" the multiplication by a within a specific modular system.

Let's find the modular inverse of $3$ modulo $7$. Possible modular inverses ${1,2,3,4,5,6}$

$$3 \cdot 1 ≡ 3 (\mod 7)$$
$$3 \cdot 2 ≡ 6 (\mod 7)$$
$$3 \cdot 3 ≡ 2 (\mod 7)$$
$$3 \cdot 4 ≡ 5 (\mod 7)$$
$$3 \cdot 5 ≡ 1 (\mod 7)$$

So, the modular inverse of $3$ modulo $7$ is $5$.

Key points to remember:

- Not every number has a modular inverse for every modulus.
- The Euclidean algorithm is a common method for finding modular inverses.
- Modular inverses are essential for solving linear congruences and other problems in modular arithmetic.

### Uniqueness of Inverses

In modular arithmetic, not all numbers have a unique inverse. This means that two different numbers can have the same inverse when working with a specific modulus.

Example:

Let's consider the modulus $7$.

The inverse of $2$ modulo $7$ is $4$ because $2 \cdot 4 ≡ 1 (\mod 7)$

The inverse of $5$ modulo $7$ is also $4$ because $5 \cdot 4 ≡ 1 (\mod 7)$

As you can see, both $2$ and $5$ have the same inverse ($4$) modulo $7$.

This <b>phenomenon</b> occurs due to the cyclical nature of modular arithmetic, where numbers wrap around within a fixed range.

### Existence of Inverses

In modular arithmetic, not all numbers have a modular inverse. The existence of an inverse depends on the relationship between the number and the modulus.

A number a has a modular inverse modulo $m$ if and only if $a$ and $m$ are coprime. This means that their greatest common divisor $gcd$ is $1$.

#### Why is coprimality important?

When $a$ and $m$ are coprime, there exists a linear combination of $a$ and $m$ that equals $1$. This linear combination can be used to find the modular inverse of $a$.

Example:

Let's consider the modulus $10$.

- The number $3$ is coprime with $10$ because their $gcd$ is $1$.

  Therefore, $3$ has a modular inverse modulo $10$.

&emsp;

- The number $5$ is not coprime with $10$ because their $gcd$ is $5$.

  Therefore, $5$ does not have a modular inverse modulo $10$.

## Extended Euclidean Algorithm

&emsp; The Extended Euclidean Algorithm is a powerful tool for calculating modular inverses. It not only finds the greatest common divisor $gcd$ of two numbers but also expresses the $gcd$ as a linear combination of them. This linear combination is crucial for determining the modular inverse.

### Delving Deeper with Examples

I want to describe this just numbers and equations.

> What is the $x$? &emsp; $ 17x ≡ 1 (\mod 43)$

First we must write $17$ and $43$ in the form of the extended Euclidean algorithm as $ax+by = c$

$$ 43 = 17 \cdot 2 + 9 $$

$$ 17 = 9 \cdot 1 + 8 $$

$$ 9 = 8 \cdot 1 + 1 $$

We have reached the remainder $1$, now let's substitute it in reverse and write it.

First we leave $1$ alone

$$ 1 = 9 - 8 \cdot 1 $$

$$ 1 = 9 - (17 - (9 \cdot 1)) \cdot 1 $$

$$ 1 = 9 - (17 - 9)$$

$$ 1 = 17 \cdot (-1) + 9 \cdot 2 $$

$$ 1= 17 \cdot (-1) + (43 - (17 \cdot 2)) \cdot 2$$

$$ 1 = 17 \cdot (-1) + 43 \cdot 2 + 17 \cdot (-4) $$

$$ 1 = 17 \cdot (-5) + 43 \cdot 2 $$

If we take $(\mod 43)$ of both sides of the equation

$$ 1 (\mod 43) = 17 \cdot (-5) $$

Wait, what is that? Isn't this equation the same as our question? We clearly see that $x$ is $-5$.

In fact, we are using the extended Euclidean algorithm in all the steps in this example and the answer we have at the end is.
