# Modular Arithmetic

What is the remainder when this expression is divided by 3? $$17 \cdot (12\cdot 19+5) -23 $$

It seems like it will take some time to calculate the result of the operation. Maybe, we choose another way to solve this. But first, we must examine modular artihmetic deeply.

<b>Definition :</b>&emsp; We say that two numbers $a$ and $b$ are congruent modulo $m$ if they have the same remainder when divided by $m$. This donete by $$ a \equiv b\mod{m} $$

In other words, a number $a$ is congruent modulo $m$ to all numbers $a+k \cdot m$ for integer $k$. In particular, if $r$ is remainder of $a$ when divided by $m$ then $a \equiv r \mod m$.

### Key Lemmas and Properties

#### Addition

- If $a \equiv b\mod{m} $, then $a + c ≡ b + c \mod m$ for any $c$.
- If $a \equiv b\mod{m} $ and $c ≡ d \mod m$, then $a + c ≡ b + d \mod m$.

#### Multiplication

- If $a \equiv b\mod{m} $, then $a \cdot c ≡ b \cdot c \mod m$ for any $c$.
- If $a \equiv b\mod{m} $ and $c ≡ d \mod m$, then $a \cdot c ≡ b \cdot d \mod m$.

These lemmas essentially state that we can perform addition and multiplication on congruent numbers without changing their congruence modulo $m$.

### Example

To find the remainder of $14 + 41 + 20 + 13 + 29$ when divided by $4$, we can replace each number with its equivalent modulo $4$:

$$ 14 ≡ 2 \mod 4, 41 ≡ 1 \mod 4, 20 ≡ 0 \mod 4, 13 ≡ 1 \mod 4, 29 ≡ 1 \mod 4$$

Then, the sum becomes:

$$2 + 1 + 0 + 1 + 1 ≡ 5 ≡ 1 \mod{4} $$

Therefore, the remainder is 1.

Now we are ready to solve the problem that asked us to calculate the remainder of when divided by $3$. $$17 \cdot (12\cdot 19+5) -23 $$

To compute the remainder, we can substitute each number by their remainder when divided by 3. As we have seen, this change does not affect the remainders of the results of arithmetic operations. This gives us

$$17 \cdot (12\cdot 19+5) -23 = 2 \cdot (0 \cdot 1 + 2) +1 ≡ 2 \mod 3 $$

This simplifies computation a lot. This way, the computer does not consume much computing power.
