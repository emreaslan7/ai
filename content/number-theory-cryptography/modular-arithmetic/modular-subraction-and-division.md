# Modular Subtraction and Division

## Modular Subtraction

Modular subtraction is a straightforward operation within the realm of modular arithmetic. It follows intuitively from the properties of modular addition.

<b>Definition:</b> &emsp; Given two integers $a$ and $b$, and $a$ modulus $m$, the modular subtraction of $a$ from $b$ modulo $m$ is defined as finding an integer $x$ such that:

$$ b - a ≡ x \mod m$$

This is equivalent to finding a number $x$ that, when added to $a$, gives $b$ modulo $m$.

#### Example:

Find $x$ such that $13 - 7 ≡ x \mod 5$. We can calculate directly: $13 - 7 = 6$. Since $6 ≡ 1 \mod 5$, we have $x = 1$.

Modular subtraction is closed under the set of integers modulo m. That is, the result of modular subtraction is always an integer between 0 and m-1.

## Modular Division

Modular division is more complex than modular subtraction and is not always defined.

<b>Definition:</b> &emsp; Given two integers $a$ and $b$, and $a$ modulus $m$, the modular division of $b$ by $a$ modulo $m$ is defined as finding an integer $x$ such that:

$$a \cdot x ≡ b \mod m$$

This $x$ is often called the modular multiplicative inverse of $a$ modulo $m$.

<b>Not Always Defined</b>

Unlike modular addition and subtraction, modular division is not always defined. For example, consider the equation $$2 \cdot x ≡ 1 \mod 4$$ There is no integer $x$ that satisfies this equation.

Is there a way when the division is possible and when it is not? We will figure this out soon!

The things turn out to be rather complicated. On the one hand, it is bad: it is nice when the computations and calculations are simple. On the other hand, this is good: complicated things are crucial for cryptography.
