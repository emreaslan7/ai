# Diophantine Theorem

&emsp; Diophantine equations, named after the ancient Greek mathematician Diophantus of Alexandria, are polynomial equations with integer coefficients that seek integer solutions.

### A Simple Example: The Candy Problem

&emsp; You have a \$$5$ bill and want to buy \$$2$ candies. The cashier only has \$$3$ coins. How many \$$5$ bills should you give, and how many \$$3$ coins will you receive in return?

We can represent this situation with the equation: $$5x=2y+5$$

where $x$ is the number of \$$5$ bills and $y$ is the number of \$$3$ coins.

A solution to this equation is $x = 3$ and $y = 5$, meaning you give the cashier $3$ \$$5$ bills and receive $5$ \$$3$ coins in change.

### The Case of $187x + 55y = 121$

&emsp; This equation is more complex, but it still follows the same principle of seeking integer solutions. One solution is $x = 3$ and $y = -8$.

However, what's intriguing is that this equation has <b>infinitely many solutions</b>. This means there are countless pairs of $x$ and $y$ that satisfy the equation.

#### The Puzzle of Solvability

&emsp; Not all Diophantine equations have solutions. The equation $187x + 55y = 45$, for instance, has no integer solutions. This begs the question: What factors determine whether a Diophantine equation has a solution?

Now we will explorer how we can solve this theorem with the help of the Euclid algorithm.

## Solving Diophantine Equations with Euclid's Algorithm

The simplest form of a Diophantine equation is the linear equation: $$ax + by = c$$

where $a$, $b$, and $c$ are integers. A fundamental theorem states that this equation has a solution if and only if the greatest common divisor of $a$ and $b$ divides $c$.

The extended Euclidean algorithm is a powerful tool for finding solutions to linear Diophantine equations. This algorithm not only computes the greatest common divisor of two integers but also finds coefficients that express the greatest common divisor as a linear combination of the two integers.

Let's we see this in example:

### Example: &emsp; $10x + 6y = 14$

<b>Do not forget!</b> The main porpuse ise finding $x$ and $y$.

1. <b>Find the gcd:</b>&emsp; $$gcd(10, 6) = 2$$

2. <b>Use the extended Euclidean algorithm:</b>&emsp; $$gcd(10,6) = 2 = 10 \cdot (-1) + 6 \cdot 2$$

3. <b>Multiply by c:</b>&emsp; To get $10x + 6y = 14$, we multiply both sides of the previous equation by 7, obtaining $$14= 10 \cdot (-1) \cdot 7 + 6 \cdot 2 \cdot 7$$ $$14= 10 \cdot (-7) + 6 \cdot 14$$ So, one solution is $$x = -7 , y = 14$$

4. <b>Find all solutions:</b>&emsp; All solutions can be expressed as $$x = -7 + 10t$$$$y = 14 - 6t$$ where $t$ is any integer.

Consider the equation $$4x + 6y = 9$$

Here, the greatest common divisor $gcd$ of $4$ and $6$ is $2$.

However, $2$ does not divide $9$ evenly. Therefore, this equation has no integer solutions.

To determine if a Diophantine equation has a solution, we find the greatest common divisor of the coefficients of the variables. If this greatest common divisor does not divide the constant term evenly, then the equation has no integer solutions.
