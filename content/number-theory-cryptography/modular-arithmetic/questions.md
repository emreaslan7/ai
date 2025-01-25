# Questions

<details style="cursor: pointer;">
<summary>
  <b>Question-1</b> :&emsp; What is the remainder of $ (77+95 \cdot 79) \cdot 82 $ when divided by $3$?

 <p>See answer ▼</p>

  </summary>

> <b>Answer</b> : 1
>
> We can substitute each number by a congruent modulo $3$ and the remainder will remain the same. Substituting each number by $ 0,1 $ or $-1$ we get $$ (-1 + (-1) \cdot 1) \cdot 1 ≡ (-2) \cdot 1 ≡ 1 \mod 3 $$

</details>

<br>
<br>

<details style="cursor: pointer;">
<summary>
  <b>Question-2</b> :&emsp; What is the remainder of $(34 -14  \cdot 25) \cdot (23 \cdot 18 + 87)$ when divided by $5$?

 <p>See answer ▼</p>

  </summary>

> <b>Answer</b> : 4
>
> We can substitute each number by a congruent modulo 5 and the remainder will remain the same. Substituting each number by $0, 1, 2, − 1$ or $−2$ we get $$ (-1 - (-1) \cdot 0) \cdot ((-2) \cdot (-2) +2 ) ≡ (-1) \cdot 6 ≡ -6 ≡ 4 \mod 5 $$

</details>

### <b>Question-3</b> : &emsp; What are the last two digits of the number $99^{99}$?

The number $99^{99}$ is huge, we do not want to compute it. Instead, we can use remainders. Note that the number consisting of the last two digits is just the remainder after the division by $100$. Thus, we are actually interested in the remainder of the number $99^{99}$ when divided by $100$.

Note that $99≡−1 \mod 100$. This allows us to simplify the computation a lot: $$99^{99} ≡ (-1)^{99} ≡ -1 ≡ 99 \mod 100 $$

Thus, the remainder of $99^{99}$ when divided by $100$ is $99$ and these are the two last digits of $99^{99}$.

In the computation above we used that $99^{99}$ is just $99$ multiplied by itself $99$ times. We then use that in the multiplication modulo some number we can substitute the numbers by their congruent.

Note that one cannot just substitute any number in any expression by its congruent. For example, we cannot substitute $99^{99}$ by $(-1)^{-1}$ We have only proved that we can substitute numbers by their congruents in additions and multiplications, and we should be careful to use only these properties.

<br>
<br>

### <b>Question-4</b> : &emsp; Is the number $3475$ divisible by $3$?

To solve this problem it is enough to compute the remainder of the number after the division by $3$: the number is divisible by $3$ if the remainder is $0$.

But how can we compute the remainder fast? We can try the same approach we used for divisibility tests. Observe that $$ 3475=3000+400+70+5 = 3 \cdot 10^{3} + 4\cdot 10^{2} + 7\cdot 10 +5$$

Now we represented our number as an arithmetic expression and we can use modular arithmetic. Note that $$ 10 ≡ 1 \mod 3 $$

Thus, for any positive $k$ $$ 10^{k} ≡ 1^{k} ≡ 1 \mod 3$$

Applying this to our number, we get $$ 3475 ≡ 3 \cdot 10^{3} + 4\cdot 10^{2} + 7\cdot 10 +5≡ 3+4+7+5 \mod 3$$

Now $$ 3+4+7+5 ≡ 19 ≡ 1 \mod 3 $$

Thus, the remainder of $3475$ when divided by $3$ is $1$ and $3475$ is not divisible by $3$.

Note, that the same argument can be applied in a general setting, giving the following lemma.

<br>
<br>

<details style="cursor: pointer;">
<summary>
  <b>Question-5</b> : &emsp; What is the remainder of the number $12^{100}$ when divided by $11$?

 <p>See answer ▼</p>

  </summary>

> <b>Answer</b> : 1
>
> we are interested in the remainder of the product of $12$ by itself $100$ times. Since congruence is preserved under multiplication, we can substitute each number $12$ by a congruent number. Note that $ 12≡1 \mod11$, so we have $ 12^{100}≡ 1^{100} ≡ 1 \mod 11$

</details>

<br>
<br>

### <b>Question-6</b> : &emsp; Find a remainder $x$ such that $3 \cdot x ≡ 5 \mod 7$?

We must multiply $3$ by such an integer that when we divide the resulting number by $7$, $5$ gives the remainder.

The first positive number that gives a remainder of $5$ is $5$. However, we cannot get $5$ by multiplying a positive number by $3$. Let's try the number sequence starting from $5$ and increasing by $7$. $$5,12,19,26,33,40...$$ the quotient of every number divisible by $3$ in this group is our answer. The smallest correct answer for $x$ is $4$.

<br>
<br>

### <b>Question-7</b> : &emsp; Find a remainder $x$ such that $12 \cdot x ≡ 3 \mod 7$?

Again, try the way just before. But first of all, we can minimize this problem to $$ 5\cdot x ≡ 3 \mod 7$$

Now, We must multiply $5$ by such an integer that when we divide the resulting number by $7$, $3$ gives the remainder. We create possible group like this: $$3,10,17,24,31,38,45,52...$$
However, we cannot get $3$ by multiplying a positive number by $5$. $10$ and $45$ are the numbers divisible by $5$ at the beginning of the group and these constitute our answers. The smallest $x$ value will be $2$.

Of course, we can create correct answer group like this: $$2,9,16,23...$$
