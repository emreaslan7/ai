# Randomness Generation

&emsp; Randomness plays a crucial role in cryptography and number theory, especially for creating secure systems. Random numbers are used for tasks like key generation, nonces, and encryption schemes. However, generating true randomness is not always easy, so we rely on **pseudorandom number generators (PRNGs)**.

## What is Randomness?

&emsp; Randomness means unpredictability. A random process produces outcomes that cannot be easily guessed. For example, flipping a coin or rolling a dice generates random results.

## Random Seed

&emsp; A **random seed** is the initial value used to start a PRNG. It acts like the "_starting point_" for generating a sequence of numbers. If the same seed is used, the PRNG will produce the same sequence. This is useful for reproducibility, such as debugging software.

Example:

- Seed = $42$
- PRNG output: $7, 12, 19, 4$

If we use Seed = $42$ again, we’ll get the same output: $7, 12, 19, 4$.

## Pseudorandom Number Generator (PRNG)

&emsp; A PRNG is an algorithm that creates a sequence of numbers that seem random but are determined by the seed. These numbers are pseudorandom because they are predictable if the algorithm and seed are known. PRNGs are fast and widely used in cryptographic systems, but they are not suitable for highly secure applications unless enhanced.

Example:

- PRNG generates numbers for password generators, simulations, or video games.

## True Randomness vs. Pseudorandomness

&emsp; True randomness comes from physical processes, like radioactive decay or thermal noise. These sources are unpredictable and provide higher security. However, they are slower and harder to implement compared to PRNGs.

---

<br/>

In summary, randomness generation, whether true or pseudorandom, is essential for creating secure cryptographic systems. Understanding the basics of seeds and PRNGs helps us design better algorithms for encryption and security.

<br/>
<br/>
<br/>
