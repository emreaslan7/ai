# Factoring

&emsp; Every integer $m>1$ can be represented as the product of primes: there exists a positive integer $k$ and primes $p_1,…,p_k$ such that $m=p_1…p_k$.

## Existence

#### What happens if $m$ is prime?

In this case, $k=1$ and $p_1=m$. So the phrase "<b>product of primes</b>" should not be taken too literally: in this case there is only one (prime) number in the "product" and the product is equal to this number. (One could even allow $m=1$ by considering an "<b>empty product</b>" with no factors that is equal to $1$, but we will not go that far.)

Note also that we do not require the $p_1,…,p_k$ to all be distinct: for example, $$ 4=2⋅2 \qquad 12=3⋅2⋅2 $$

### Prove the Theorem

&emsp; Let us look at this statement from the algorithmic point of view: given some $m>1$, how can we find a prime decomposition (existence of which we want to prove)?

If $m$ is prime, we already know that this decomposition consists of $m$ only. What if $m$ is composite (not prime)?

By definition then, $m=uv$ for some $u,v$ where $1<u,v<m$. How can we then find the decomposition of $m$?

It is easy: &emsp;just combine the decompositions of $u$ and $v$, forming a long product from the two shorter ones. Both $u$ and $v$ are smaller that $m$, so we may assume (by induction in our proof, or a recursive call in our algorithm) that for $u$ and $v$ the decompositions exist and can be found.

```python
# Finds the minimal divisor>1 of the given integer m>1
def min_divisor(m):
    for d in range(2, m + 1):
        if m % d == 0:
            return d
        # optimization:
        if d * d > m:
            return m


def is_prime(m):
    return m == min_divisor(m)


def factoring(m):
    if is_prime(m):
        return [m]
    else:
        divisor = min_divisor(m)
        factors = factoring(m // divisor)
        factors.append(divisor)
        return factors
```

&emsp; The numbers of the form $2^{2^n} +1$ were considered long ago by Pierre Fermat (1607--1665, [Wikipedia](https://en.wikipedia.org/wiki/Pierre_de_Fermat)), famous for the Last Fermat theorem; we will see the other result of him later in this chapter. He noticed that $ \quad 2^0 , 2^1 +1 , 2^2 +1 , 2^4 +1 , 2^8 +1 , 2^{16} +1 \quad $ are all primes and conjectured that all subsequent numbers are also prime.

Only later Leonhard Euler (1707--1783,[Wikipedia](https://en.wikipedia.org/wiki/Leonhard_Euler)) discovered the factorization of $ \quad 2^{32}+1 \quad$ in 1732, and more than a century after that was needed to find the factorization of $ \quad 2^{64} +1 \quad$. Now our simple program gives these factorization in seconds. (But do not forget two optimization lines, otherwise it would take much longer.)

For the next [Fermat numbers](https://en.wikipedia.org/wiki/Fermat_number) one should use more advanced factorization algorithms. There is a module <b>sympy</b> in python that provides function <b>factorint</b> that factors $ \quad 2^{128}+1 \quad$ and $\quad 2^{256}+1 \quad$ in minutes, but Fermat numbers grow really fast (and only few more factorizations are known, even if we use the most advanced factorization tools).

## Unique Factoring

Decomposition of an integer $m>1$ into a product of prime factors is essentially unique. The word <b>essentially</b> here means that we ignore the ordering of the factors: they can be permuted in any way and we still get the same decomposition. For example, $$12=2⋅2⋅3=2⋅3⋅2=3⋅2⋅2$$

First, let us note that we can group identical factors in the decomposition. In this we get a product of the form
$$ m=p_1^{n_1}p_2^{n_2}...p_k^{n_k} $$

Here all $p_i$ are different, and all $n_i$ are positive integers (the number of occurrences of $p_i$ in the factorization).

The number $n_i$ is called the multiplicity of a prime $p_i$ in $m$. If $p_i$ does not appear in the decomposition, we say that the multiplicity of $p_i$ in $m$ equals zero. (This sounds natural since we may add fictional term $ \quad p_i^{0} = 1 $ in the factoring.)

### Divisors and Multiplicity

Complete the statement: an integer $d>1$ is a divisor of an integer $m>1$ if and only if the multiplicity of $p⟨…⟩$ for every prime $p$.

The last part can be filled as follows: if the multiplicity of $p$ in $d$ does not exceed the multiplicity of $p$ in $m$, for every prime $p$. Indeed, imagine that $d$ is a divisor of $m$, i.e., that $m=dq$ for some $q$. Then the (unique) factorization of $m$ is obtained by concatenating the factorizations of $d$ and $q$. Therefore, the multiplicity of every prime $p$ in $m$ is the sum of its multiplicities in $d$ and $q$, and is greater or equal than the multiplicity of $p$ in $d$.

On the other hand, if every prime appears in the factorization of $m$ at least as many times as for $d$, then we can add missing factors to $d$ to obtain $m$. If $q$ is the product of these missing factors, then $dq=m$, so $d$ divides $m$.

##### Problem

Consider all the positive divisors of $ 2^4 3^3 = 432$. How many of them do exist (including <b>trivial divisiors</b> $1$ and $432$)?

As the previous problem shows, all the divisors have the form $ 2^k 3^l $ where $0≤k≤4$ and $0≤l≤3$. For $k=l=0$ we get $1$, for $k=4$ and $l=3$ we get $432$. We may list all of them systematically:
$$ 2^0 \cdot 3^0 = 1 $$ $$ 2^0 \cdot 3^1 = 3 $$ $$ 2^0 \cdot 3^2 = 9 $$ $$ 2^0 \cdot 3^3 = 27 $$ $$ 2^0 \cdot 3^4 = 81 $$ $$ 2^1 \cdot 3^0 = 2 $$ $$ 2^1 \cdot 3^1 = 6, ...$$

We do not need to write them all explicitly; we have only to count them. Each of five powers of $2$ (i.e., $2^0, 2^1 , 2^2 , 2^3 , 2^4 $) is combined with four powers of $3$ (i.e., $3^0,3^1,3^2,3^3$), so we get $ 5 × 4 =20$ combinations that lead (as we know) to $20$ divisors.

<br />

<details style="cursor: pointer;">
<summary>
  <b>Question</b> :&emsp; How many positive divisors has $ 2^{10} × 3^{15} × 5^{20} $?

 <p>See answer ▼</p>

  </summary>

> <b>Answer</b> : 3696
>
> $$ 11 × 16 × 21 = 3696 $$

</details>

<br/>
<br/>

<details style="cursor: pointer;">
<summary>
  <b>Question</b> :&emsp; What is the minimal positive number that has exactly $15$ divisors?

 <p>See answer ▼</p>

  </summary>

> <b>Answer</b> : 144
>
> We must choose the smallest prime numbers and large $n$ numbers for this.
> $$ 2^4 × 3^2 = 144 $$

</details>
