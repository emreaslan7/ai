# RSA Cryptosystem

## What is the RSA?

&emsp; The RSA algorithm is an asymmetric cryptography algorithm widely regarded as the most secure encryption method. You may wonder what I mean by saying the RSA algorithm is asymmetric. It means it works on two different keys: the Public Key and the Private Key. And, as the name implies, the Public Key is distributed to everyone while the Private Key is kept private. The RSA algorithm is based on how difficult it is to factorize a large integer.

The RSA algorithm is named after those who invented it in 1978: Ron Rivest, Adi Shamir, and Leonard Adleman. [(whitepaper)](https://people.csail.mit.edu/rivest/Rsapaper.pdf)

The following illustration highlights how asymmetric cryptography works:

<div style="display: flex; justify-content: center;">
<img src="../../img/number-theory-cryptography/rsa-01.svg" />
</div>

## How does the RSA algorithm work?

RSA algorithm comprises four stages, and those four stages are:

1. **Key generation**: To generate a private key (to keep) and a public key (to share).

2. **Key distribution**: Flood the network with the public key.

3. **Encryption**: The sender encrypts the message using the receiver’s public key.

4. **Decryption**: The message is decrypted by the receiver using its private key.

### 1. Key Generation

The key generation process is as follows:

1. **Select two prime numbers**: Select two large prime numbers, say $p$ and $q$.

2. **Calculate $n$**: Calculate $n$, where $$n = p \times q$$

3. **Calculate $\phi(n)$**: Calculate Euler's totient function of $n$, where $$\phi(n) = (p-1)(q-1)$$

4. **Select $e$**: Select an integer $e$ such that $1 < e < \phi(n)$ and $e$ is co-prime to $\phi(n)$.

5. **Calculate $d$**: Calculate $d$, the modular multiplicative inverse of $e$ modulo $\phi(n)$. Calculate $d$ such that $$e.d=1 \mod ϕ(n)$$
   $d$ can be found using the _extended euclidean algorithm_.

6. **Public key**: The public key is $(n, e)$.

7. **Private key**: The private key is $(n, d)$.

### 2. Key Distribution

The public key is distributed, and the private key is kept secret.

### 3. Encryption

The encryption process is as follows:

1. **Convert the plaintext to a number**: Convert the plaintext to a number $m$.

2. **Encrypt the message**: Compute the ciphertext $c$ from the plaintext $m$ using the public key $(n, e)$, where $$c = m^e \mod n$$

### 4. Decryption

The decryption process is as follows:

1. **Decrypt the message**: Compute the plaintext $m$ from the ciphertext $c$ using the private key $(n, d)$, where $$m = c^d \mod n$$

## Proof

&emsp; Since $d$ is the inverse of $e$ modulo $φ(n)$, there exists an integer $k$ such that $de=kφ(n)+1$. Assuming that $gcd(m,n)=1$, we have that
$m^{φ(n)} ≡1 \mod n$, by Euler's Totient Theorem. Thus, $$(m^e)^d ≡ m^{ed} ≡ m^{kφ(n)+1} ≡ (1)^k m ≡m \mod n$$ In case $\gcd(m,n)>1$, we know that
$m$ is divisible by either $p$ or $q$. It cannot be divisible by both, since $m<n$. Assume, without loss of generality, that $p$ divides $m$. Then,
$\gcd(m,q)=1$. In this case, $m^{ed} ≡ 0 \mod p$ By the argument above, $m^{ed} ≡ m \mod q$. Since, $m<n=pq$, we conclude that $m^{ed} ≡ m \mod n$, by the Chinese Reminder Theorem.

---

## Example

- Choose $p = 3 \space$ and $\space q = 11$.

- Compute $$n = p \times q \space$$ $$\space 33 =  3 \times 11$$

- Compute $$φ(n) = (p - 1) \times (q - 1)$$ $$20 = 2 \times 10$$

- Choose e such that $1 < e < φ(n)$ and $e$ and $φ (n)$ are coprime. Let $e = 7$

- Compute a value for d such that $$d.e \mod φ(n) = 1$$ One solution is $$(3.7) \mod 20 = 1$$

- Public key is $(e, n) => (7, 33)$

- Private key is $(d, n) => (3, 33)$

- The encryption of $m = 2$ is $$c = m^e \mod n$$ $$2^7 \mod 33$$ $$c=29$$

- The decryption of $c = 29$ is $$m = c^d \mod n$$ $$m = 29^3 \mod 33$$ $$m=2$$

---

## Pseudocode

```python
p = 3
q = 11

n = x * y
# n = 33.

# compute the totient, phi
int phi = (x-1)*(y-1)
# phi = 20.

int e = findCoprime(phi)
# find an 'e' which is > 1 and is a co-prime of phi.
# e = 7 satisfies the current values.

# Using the extended euclidean algorithm, find 'd' which satisfies
# this equation:
d = (1 mod (phi))/e
# d = 3 for the example values.

public_key = (e=7, n=33)
private_key = (d=3, n=33)

# Given the plaintext P=2, the ciphertext C is :
C = (2^7) % 33 = 29
# To decrypt the cypher text C:
P = (29^3) % 33 = 2
```

## Security of RSA

&emsp; The security of RSA relies on the assumption that decoding a message without the private key is computationally difficult, though no formal proof guarantees this.

A common attack involves factoring the public modulus $n$ into its prime components $p$ and $q$. Once $n$ is factored, the private key $d$ can be computed as the modular inverse of $e$ modulo $\phi(n)$, allowing decryption of any ciphertext $ c \equiv m^e \mod n $.

This approach is impractical because factoring large integers (e.g., $1000$ digits) is computationally infeasible. RSA's security therefore relies on the hardness of factoring.

&emsp; Another potential attack is solving the **modular root problem**, where $m$ is recovered directly from $m^e \mod n$ without factorization. While inefficient methods exist, no efficient solution has been discovered, making this problem another cornerstone of RSA's security.

<br/>
<br/>
