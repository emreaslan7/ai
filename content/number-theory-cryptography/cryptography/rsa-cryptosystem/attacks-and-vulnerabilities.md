# Attacks and Vulnerabilities

&emsp; RSA is considered secure. But this only means that we don't know an efficient decoding algorithm that works for all messages. In this section, we will study (and implement!) a few attacks that work well in certain special cases.

## Small Space of Messages

Alice needs to send one of the following messages to Bob:

**attack**, or **retreat**.

This essentially conveys one bit of information. To simplify, Alice and Bob agree to represent these messages as:

$m = 0$ for **attack**, and $m = 1$ for **retreat**.

However, this approach is insecure. Why?

If Alice sends a ciphertext $𝑐 ≡ 𝑚^𝑒 \mod 𝑛$ to Bob, an eavesdropper (Eve) can encode both $m=0$ and $m=1$ using Bob’s public key $(n,e)$, then compare the results to $𝑐$. This is because cryptographic conventions often assume that the mapping of messages to integers is public. Even if Eve doesn’t initially know what the bit represents, she can intercept future ciphertexts and recognize if they are identical to previous ones.

This attack extends to cases where small integers other than $0$ and $1$ are used, or when the total number of possible messages is small. In such cases, Eve can enumerate all potential encodings and match them to the intercepted ciphertext.

---

### Defense Mechanism

To counteract this vulnerability, the space of possible messages must be artificially expanded, making it computationally infeasible for Eve to enumerate all possibilities. For instance, Alice can append $128$ random bits to her $128$-bit message before encryption. After decoding, Bob discards these extra random bits. This makes it practically impossible for Eve to guess among $2^{128}$ possibilities within a reasonable timeframe.

### Problem Statement

Alice, a secret agent, has sent one of these messages to the center:

**attack** , **don’t attack**, or **wait**.

She encrypted the message using the public key
$(modulo,exponent)$, which is accessible to you. You intercepted her ciphertext and want to determine the content of her message.

You are given access to two functions:

```
Encrypt(message, modulo, exponent)
```

Encrypts a message using RSA with the provided public key, returning the ciphertext as a large integer.

```
DecipherSimple(ciphertext, modulo, exponent, potential_messages)
```

This function needs to be implemented. It should:

- Take the ciphertext, public key, and potential messages.
- Return the decrypted message as a string.

_For example_:

If Alice encrypts "**wait**" with the given key and produces the ciphertext $139763215$, your implementation of DecipherSimple should return "**wait**" when provided with the ciphertext, public key, and the list of potential messages ["**attack**", "**don’t attack**", "**wait**"].

```python
def DecipherSimple(ciphertext, modulo, exponent, potential_messages):
    # Fix this implementation
    if ciphertext == Encrypt(potential_messages[0], modulo, exponent):
        return potential_messages[0]
    return "don't know"


modulo = 101
exponent = 12
ciphertext = Encrypt("attack", modulo, exponent)
print(ciphertext)
print(DecipherSimple(ciphertext, modulo, exponent,
                      ["attack", "don't attack", "wait"]))
```

## Small Prime Problem

When Bob prepares his RSA keys, he generates two random prime numbers, $p$ and $𝑞$. Let’s assume one of these primes, say $p$, is small ($p≤10^6$). While this is a rare case in real-world cryptography, it highlights a potential vulnerability.

Why is this insecure?

If $𝑝$ is small, an attacker like Eve can factorize $n$ (where $n=p⋅q$) quickly. She can:

1. Enumerate all primes up to $10^6$.
2. Check which prime divides $n$.

Once Eve finds $p$, she calculates $q=n/p$. With both $𝑝$ and $q$ known, Eve can derive the private key and decrypt any message encrypted with $𝑛$.

### Defense Strategy

Bob should generate both $𝑝$ and $q$ such that they are sufficiently large — e.g., $4096$-bit primes. Enumerating all primes of this size would be computationally infeasible for any attacker.

### Problem Statement

Alice is using RSA encryption with a public key (modulo,exponent), where:

$$ \text{modulo} = p \cdot q $$

One of the primes, either $p$ or $𝑞$, is less than $10^6$. You intercepted her ciphertext and need to decrypt it.

Given Functions

> Decrypt(ciphertext, p, q, e)

This function decrypts the ciphertext using the private key ($p,q$) and the public exponent ($e$).

> DecipherSmallPrime(ciphertext, modulo, exponent)

This function must be implemented to:

- Factorize $modulo$ by finding a small prime $𝑝$.
- Use $𝑝$ and $q$ to decrypt the ciphertext.

#### Implementation Steps

To solve the problem, the implementation of DecipherSmallPrime should:

1. Iterate through all primes up to $10^6$.

2. For each prime $𝑝$, check if $modulo \mod p=0$.

3. Once $𝑝$ is found, compute $q=modulo/p$.

4. Use $p$, $𝑞$, and the provided Decrypt function to decrypt the ciphertext.

5. Return the decrypted message as a string.

```python
def DecipherSmallPrime(ciphertext, modulo, exponent):
    if modulo % 2 == 0:
        small_prime = 2
        big_prime = modulo // 2
        return Decrypt(ciphertext, small_prime, big_prime, exponent)
    return "don't know"


modulo = 101 * 18298970732541109011012304219376080251334480295537316123696052970419466495220522723330315111017831737980079504337868198011077274303193766040393009648852841770668239779097280026631944319501437547002412556176186750790476901358334138818777298389724049250700606462316428106882097210008142941838672676714188593227684360287806974345181893018133710957167334490627178666071809992955566020058374505477745993383434501768887090900283569055646901291270870833498474402084748161755197005050874785474707550376333429671113753137201128897550014524209754619355308207537703754006699795711188492048286436285518105948050401762394690148387
exponent = 239
ciphertext = Encrypt("attack", modulo, exponent)
print(ciphertext)
print(DecipherSmallPrime(ciphertext, modulo, exponent))
```

## Small Difference of Primes

When preparing the keys, Bob generates two random prime numbers, $p$ and $q$. They turn out to be close to each other: say, $∣p−q∣<10^6$. (As with the previous example, the probability of this event is negligible.)

### Defense Strategy

Bob should verify that $∣p−q∣$ is large enough after generating the primes. If the difference is small, he should regenerate the primes to avoid this vulnerability.

### Problem Statement

Alice is using RSA encryption with a public key (modulo,exponent) such that $modulo=p⋅q$ with $∣p−q∣<5000$, and you know about it. You want to break the cipher and decrypt her message.

Given Functions

> Decrypt(ciphertext,p,q,e)

which decrypts the **ciphertext** given the private key $p,q$ and the public exponent $e$. You also have access to the function $IntSqrt(n)$ which takes integer $n$ and returns the largest integer $x$ such that $x^2≤ n$ . You are also given the function

> DecipherSmallDiff(ciphertext,modulo,exponent)

and you need to fix its implementation so that it can decipher the **ciphertext** in case when the difference between prime factors of the public modulo is smaller than $5000$.

```python
def DecipherSmallDiff(ciphertext, modulo, exponent):
    small_prime = IntSqrt(modulo)
    big_prime = modulo // small_prime
    return Decrypt(ciphertext, small_prime, big_prime, exponent)


p = 1000000007
q = 1000000009
n = p * q
e = 239
ciphertext = Encrypt("attack", n, e)
message = DecipherSmallDiff(ciphertext, n, e)
print(ciphertext)
print(message)def DecipherSmallDiff(ciphertext, modulo, exponent):
    small_prime = IntSqrt(modulo)
    big_prime = modulo // small_prime
    return Decrypt(ciphertext, small_prime, big_prime, exponent)


p = 1000000007
q = 1000000009
n = p * q
e = 239
ciphertext = Encrypt("attack", n, e)
message = DecipherSmallDiff(ciphertext, n, e)
print(ciphertext)
print(message)
```

## Insufficient Randomness

Every day, Bob generates two fresh random prime numbers $p$ and $q$ and then publishes $n=pq$ as part of the public key. To do this, Bob uses a random number generator. Assume that for some two days, the same prime $p$ was used: $n_1= pq_1$ and $n_2 = pq_2$.

### Problem Statement

Alice and Angelina generated their RSA keys using insufficient randomness, which caused the same prime $𝑝$ to be used in their public keys:

$$modulo_1 = 𝑝 ⋅ 𝑞_1 \space \space \space \space \space modulo_2 = 𝑝⋅𝑞_2$$

If the public modulo $modulo_1$ and $modulo_2$ share a common prime factor $p$, we can efficiently compute $p$ and use it to decrypt both ciphertexts.

> DecipherCommonDivisor(first_ciphertext, first_modulo, first_exponent, second_ciphertext, second_modulo, second_exponent)

Steps:

1. Compute $𝑝=GCD(first_modulo,second_modulo)$.
2. Calculate $𝑞_1= first_modulo / 𝑝$ and $𝑞_2 =second_modulo / 𝑝$.
3. Decrypt both ciphertexts using the provided Decrypt function:

   - Decrypt the first ciphertext using $𝑝$ and $𝑞_1$.
   - Decrypt the second ciphertext using $p$ and $𝑞_2$.

4. Return the two decrypted messages.

```python

def GCD(a, b):
    if b == 0:
        return a
    return GCD(b, a % b)


def DecipherCommonDivisor(first_ciphertext, first_modulo, first_exponent,
                          second_ciphertext, second_modulo, second_exponent):
    # Fix this implementation to correctly decipher both messages in case
    # first_modulo and second_modulo share a prime factor, and return
    # a pair (first_message, second_message). The implementation below won't work
    # if the common_prime is bigger than 1000000.
    for common_prime in range(2, 1000000):
        if first_modulo % common_prime == 0 and second_modulo % common_prime == 0:
            q1 = first_modulo // common_prime
            q2 = second_modulo // common_prime
            return (Decrypt(first_ciphertext, common_prime, q1, first_exponent),
                    Decrypt(second_ciphertext, common_prime, q2, second_exponent))
    return ("unknown message 1", "unknown message 2")


# Example usage with common prime p and different second primes q1 and q2
p = 101
q1 = 18298970732541109011012304219376080251334480295537316123696052970419466495220522723330315111017831737980079504337868198011077274303193766040393009648852841770668239779097280026631944319501437547002412556176186750790476901358334138818777298389724049250700606462316428106882097210008142941838672676714188593227684360287806974345181893018133710957167334490627178666071809992955566020058374505477745993383434501768887090900283569055646901291270870833498474402084748161755197005050874785474707550376333429671113753137201128897550014524209754619355308207537703754006699795711188492048286436285518105948050401762394690148387
q2 = 1000000007
first_modulo = p * q1
second_modulo = p * q2
first_exponent = 239
second_exponent = 17
first_ciphertext = Encrypt("attack", first_modulo, first_exponent)
second_ciphertext = Encrypt("wait", second_modulo, second_exponent)
print(DecipherCommonDivisor(first_ciphertext, first_modulo, first_exponent,
                            second_ciphertext, second_modulo, second_exponent))

```

<br/>
<br/>
<br/>
<br/>
