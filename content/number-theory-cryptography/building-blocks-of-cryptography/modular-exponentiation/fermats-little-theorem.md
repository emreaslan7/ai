# Fermat's Little Theorem

## Introduction

&emsp; Fermat's Little Theorem, a fundamental result in number theory, provides a powerful tool for understanding the behavior of integers modulo a prime number. It has far-reaching implications in various fields, including cryptography, computer science, and pure mathematics.

## The Theorem

Fermat's Little Theorem states that if $p$ is a prime number and $a$ is an integer not divisible by $p$, then: $$a^{p−1} ≡ 1 \mod p$$

In other words, if we raise $a$ to the power of $p−1$ and divide the result by $p$, the remainder will always be $1$.

## Proof

&emsp; To prove Fermat's Little Theorem, we will consider the set of integers ${1,2,3,...,p−1}$. We will multiply each element of this set by $a$ modulo $p$:

$$a,2a,3a,...,(p−1)a$$

No two of these numbers are congruent modulo $p$. To see why, suppose that $ia≡ja(modp)$ for some $1≤i,j≤p−1$. Then, $p$ would divide $(i−j)a$. Since $p$ does not divide $a$, it must divide $i−j$. However, $∣i−j∣<p$, so this is impossible.

Therefore, the set ${a,2a,3a,...,(p−1)a}$ is a rearrangement of the set ${1,2,3,...,p−1}$ modulo $p$. Multiplying all the elements of these sets together, we get:

$$a⋅2a⋅3a⋅...⋅(p−1)a≡1⋅2⋅3⋅...⋅(p−1)(modp)$$

Simplifying, we obtain:

$$a^{p−1}(p−1)!≡(p−1)! \mod p$$

Since $(p−1)!$ is not divisible by $p$, we can cancel it from both sides, leaving us with:

$$a^{p−1} ≡1 \mod p$$

This completes the proof of Fermat's Little Theorem.

## Problems and Solutions

#### Problem 1:

Find the remainder when $2^{100}$ is divided by $11$.

#### Solution:

Since $11$ is a prime number, we can apply Fermat's Little Theorem:

$2^{10}≡1(mod11)$

Therefore, $$ 2^{100} = (2^{10})^{10} \equiv 1^{10} \equiv 1 \pmod{11} $$

So, the remainder when $2^{100}$ is divided by $11$ is $1$.

---

#### Problem 2:

Find the last digit of $3^{1001}$.

#### Solution:

The last digit of a number is equivalent to its remainder when divided by $10$. So, we need to find $3^{1001} \mod 10$.

We can not use Fermat's Theorem in this problem. Because $p$ is not prime. Think another way:

Note that $3^4≡1 \pmod 10$. Therefore, $$ 3^{1001} = 3^{4\cdot 250 + 1} \equiv (3^4)^{250} \ 3 \equiv 1^{250} \ 3 \equiv 3 \pmod{10} $$

Hence, the last digit of $3^{1001}$ is $3$.

## Conclusion

&emsp; Fermat's Little Theorem is a powerful tool with wide-ranging applications in number theory and computer science. Its elegant proof and practical utility make it a cornerstone of modern mathematics.

As we continue to explore the depths of number theory, Fermat's Little Theorem will undoubtedly remain a valuable asset in our mathematical toolkit.
