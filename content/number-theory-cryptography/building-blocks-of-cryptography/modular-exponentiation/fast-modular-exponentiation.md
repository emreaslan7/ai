# Fast Modular Exponentiation

&emsp; Our next results (**_Fermat's little theorem_** and its generalization, **_Euler's totient theorem_**) deal with modular exponentiation, i.e., with expressions of type $a^b\mod(c)$ where $a,b,c$ are integers. But before stating and proving this result, let us discuss the purely algorithmic question: how to compute $a^b \mod(c)$ reasonably fast?

How would you compute $314^{271}\mod(123)$ and $314159265358^{2718281828}\mod(123456789)$ ?

Computing this in python directly gives the following result:

```python
print((314 ** 271) % 123)
print((314159265358 ** 2718281828) % 123456789)

# 38
# it will be crashed...
```

If you try this code on your computer, you can see that your computer is struggling. Because we know that pc uses which methot. This method consists of taking the exponent of the number and then taking its modulo.

The right way to compute modular exponentiation in python is to use the built-in **pow** method. The following code produces the result in the blink of an eye. We will learn what is going on under the hood below.

```python
print((314 ** 271) % 123)
print((314159265358 ** 2718281828) % 123456789)

# 38
# 32073907
```

##### Problem

Compute $8^{100} \mod(63)$.

Of course, starting with computing $8^{100}$ is a bad idea. Note that $8^2 = 64 ≡ 1 \mod{63}$, and
$$ 8^{100} = 8^2 \cdot 8^2 \cdot 8^2 \text{ ... } 8^2 ≡ 1 \cdot 1 \cdot 1 \cdot 1 \text{ ... } 1 \mod(63) $$ hence the answer is $1$.

<br/>
<br/>

##### Problem

Prove that $2^{1001}+3^{1001}$ is divisible by $5$.

We know $3 ≡ (-2) \mod(5)$. If we write $-2$ instead of $3$. $$ 2^{1001}+(-2)^{1001}$$ And it is equal to zero. So, it's divisible by $5$.

<br/>
<br/>

##### Problem

Find $2^{1025} \mod(17)$.

Since $2^{1024} = 16^{256} ≡ (-1)^{256} ≡ 1 \mod(17)$, answer is $2$.

<br/>
<br/>

&emsp; Now we can compute $a \mod c$, $a^2 \mod c$, $a^3 \mod c$, ... sequentially, multiplying each time by $a$ and keeping only the remainder modulo $c$. In this way we do not need to store large numbers (only numbers smaller than $c$). Does this approach work for our example $314159265358^{2718281828}\mod(123456789)$?

The size of numbers is no more a problem, but we need to perform $2718281827$ multiplications --- too many. Can we do better? This is a general question for computing powers (that makes sense not only for modular computations).

Let $x$ be some number. Can you compute $x^{64}$ in less than 63 multiplications (that would be needed if we compute $x$, $x^2$, $x^3$, ..., $x^{64}$ sequentially)?

For this problem the answer is especially easy to guess, since $64 = 2^6$ and repeated squaring helps:
$$x^2 = x \cdot x, \quad x^4 = x^2 \cdot x^2, \quad x^8 = x^4 \cdot x^4, \quad x^{16} = x^8 \cdot x^8, \quad x^{32} = x^{16} \cdot x^{16}, \quad x^{64} = x^{32} \cdot x^{32}.$$

In this way the exponents grow faster ($\times 2$ instead of $+1$ at every step), and we use only $6$ multiplications.

In general, to compute $x^n$ for $n = 2^k$ we need $k = \log_2 n$ multiplications.

For $x^{65}$ we may multiply $x^{64}$ by $x$. Since we have also computed previous powers of 2, we may use that $x^{68} = x^{64} \cdot x^4$ and $x^{77} = x^{64} \cdot x^8 \cdot x^4 \cdot x$ (note that $77 = 64 + 8 + 4 + 1$). Similar trick can be used in the general.

```python
def fast_modular_exponentiation(b, e, m):
    assert m > 0 and e >= 0
    if e == 0:
        return 1
    if e == 1:
        return b
    if e % 2 == 0:
        return fast_modular_exponentiation((b * b) % m, e // 2, m)
    else:
        return (fast_modular_exponentiation(b, e - 1, m) * b) % m


print(fast_modular_exponentiation(123456789222, 456788, 598723))

# 32073907
```

The program implementing this idea computes $314159265358^{2718281828} \bmod 123456789$ in a fraction of a second.

<br/>
<br/>

##### Problem

Prove that the number of multiplications when computing $x^n$ (for $n \geq 1$) both in this program and in the method described above (using binary notation) is the same:
$$(\text{number of bits in } n) + (\text{number of ones in } n) - 2$$
where counting the bits and ones in $n$ we use the binary representation of $n$.

It turns out that this is not exactly the minimal number of multiplications needed to compute $x^n$. For example, for $x^{15}$ we get 6 multiplications with both our approaches, while one can compute it in 5 multiplications:

$x^2 = x \cdot x; \quad x^4 = x^2 \cdot x^2; \quad x^5 = x^4 \cdot x; \quad x^{10} = x^5 \cdot x^5; \quad x^{15} = x^{10} \cdot x^5.$

See [Wikipedia article](https://en.wikipedia.org/wiki/Addition-chain_exponentiation) about addition chains that minimize the number of multiplications.
