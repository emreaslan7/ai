# Divisibility

When we say that a number $a$ is divisible by $a$ number $b$, we mean that $a$ can be divided by $b$ without leaving a remainder. In other words, $a$ is $a$ multiple of $b$.

#### Example:

$15$ is divisible by $3$ because $15$ divided by $3$ equals $5$ without any remainder.
$15$ is not divisible by $4$ because 15 divided by $4$ leaves a remainder of $3$.

**Formal Definition:**

We can write this mathematically as:

If $a$ is divisible by $b$, we can write $a$ as $b$ multiplied by another integer $k$. In symbols: $a = b \cdot k$.

Equivalently, we can use the vertical line notation: $b \mid a$. This means b divides a.

**Example:**

$15$ is divisible by $3$ because $15 = 3 \cdot 5$. Therefore, we can also write $3 \mid 15$.

#### Checking Divisibility in Python:

In Python, we can use the modulo operator (%) to check if a number is divisible by another. If the remainder is $0$, then the first number is divisible by the second.

```python
print(15 % 3 == 0) # Output: True
print(15 % 4 == 0) # Output: False
```

### Properties of Divisibility:

- **_$lemma$_**: If $c$ divides $a$ and $b$, then $c$ divides $a \pm b$.

  **_$proof$_**: Since $c$ divides $a$, there is $k_1$ such that $a = c \cdot k_1$. Similarly, there is $k_2$ such that $b = c \cdot k_2$. Then

  $$a\pm b = c \cdot k_1 \pm c \cdot k_2 = c \cdot (k_1 \pm k_2)$$

- **_$lemma$_**: If $b \mid a$, then for any integer $c$, $b \mid (a \cdot c)$.

  **_$proof$_**: Since $b \mid a$, we have that $ a = b \cdot k$ for some $k$. We can multiply both sides of equality by $c$: $c \cdot a = b \cdot (c\cdot k)$. By defination, this means that $c \cdot a$ is divisible by $b$.

## Quiz - Division by 101

<details style="cursor: pointer;">
<summary>
  <b>Question</b> : How many $3$-digit non-negative numbers are there that have remainder $7$ when divided by 101? Here we assume that $1$-digit and $2$-digit numbers are also $3$-digit, they just start  with  $0$.
    </summary>

> <b>Answer</b> : $10$
>
> All numbers $a$ having the remainder $7$ when divided by $101$ have the form $a = 7 + q \cdot 101 $ for integer $q$. The number $a$ given by this expression is non-negative and $3$-digit for $q$ between $0$ and $9$, inclusive.

</details>

These properties are essential for understanding divisibility and proving various mathematical theorems.

As you can see, divisibility is a fundamental concept in mathematics that has practical applications in various fields, from computer science to cryptography.
