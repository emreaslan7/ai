# Reinforcement Learning

<!-- toc -->

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a machine learning paradigm where an **agent** learns to make sequential decisions by interacting with an **environment** to maximize a cumulative **reward**. Unlike supervised learning, where labeled data is provided, RL relies on trial and error, receiving feedback in the form of rewards or penalties.

### Key Characteristics of Reinforcement Learning:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/reinforcement-learning-01.png" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

- **Agent**: The entity making decisions (e.g., a robot, a self-driving car, or an AI player in a game).
- **Environment**: The external system with which the agent interacts.
- **State (s)**: A representation of the current situation of the agent in the environment.
- **Action (a)**: A choice made by the agent at a given state.
- **Reward (R)**: A numerical value given to the agent as feedback for its actions.
- **Policy ( $ \pi$ )**: A strategy that maps states to actions.
- **Return (G)**: The cumulative reward collected over time.
- **Discount Factor ( $ \gamma $ )**: A value between 0 and 1 that determines the importance of future rewards.

<br/>
<br/>

**Mars Rover Example**

Let's illustrate RL concepts using a **Mars Rover** example. Imagine a rover exploring a **1D terrain** with **six grid positions**:

Each position is numbered from 1 to 6. The rover starts at position **4**, and it can move **left (-1)** or **right (+1)**. The goal is to maximize its rewards, which are given at positions **1** and **6**:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/reinforcement-learning-02.png" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

- **Position 1 reward**: **100** (e.g., a research station with supplies)
- **Position 6 reward**: **40** (e.g., a safe resting point)
- **Other positions reward**: **0**

**States, Actions, and Rewards**

| State     | Possible Actions                | Reward |
| --------- | ------------------------------- | ------ |
| 1         | Move right (+1)                 | 100    |
| 2         | Move left (-1), Move right (+1) | 0      |
| 3         | Move left (-1), Move right (+1) | 0      |
| 4 (Start) | Move left (-1), Move right (+1) | 0      |
| 5         | Move left (-1), Move right (+1) | 0      |
| 6         | Move left (-1)                  | 40     |

- The **agent (rover)** must decide which direction to move.
- The **state** is the current position of the rover.
- The **action** is moving left or right.
- The **reward** depends on reaching the goal states (1 or 6).

<br>

**How the Rover Decides Where to Go**

The rover's decision is based on maximizing its **expected future rewards**. Since it has two possible goal positions (1 and 6), it must evaluate different strategies. The rover should consider the following:

1. **Immediate Reward Strategy**

   - If the rover focuses only on immediate rewards, it will move randomly, as most positions (except 1 and 6) have a reward of 0.
   - This strategy is **not optimal** because it doesn't take future rewards into account.

2. **Short-Term Greedy Strategy**

   - If the rover chooses the nearest reward, it will likely go to position **6** since it's closer than position **1**.
   - However, this might not be the best long-term decision.

3. **Long-Term Reward Maximization**
   - The rover must evaluate how much **discounted future reward** it can accumulate.
   - Even though position **6** has a reward of **40**, position **1** has a **much higher reward (100)**.
   - If the rover can reliably reach **position 1**, it should favor this route, even if it takes more steps.

To formalize this, the rover can compute the expected return **G** for each possible path, considering the **discount factor ($ \gamma $)**.

### Discount Factor ($ \gamma $) and Expected Return

The discount factor **$ \gamma $** determines how much future rewards are valued relative to immediate rewards. If $ \gamma = 1 $, all future rewards are considered equally important. If $ \gamma = 0.9 $, future rewards are slightly less important than immediate rewards.

For example, if the rover follows a path where it expects to reach **position 1** in 3 steps and receive **100** reward, the discounted return is:

$$
G = 100 \times \gamma^3 = 100 \times 0.9^3 = 72.9
$$

If it reaches **position 6** in 2 steps and receives **40** reward, the return is:

$$
G = 40 \times \gamma^2 = 40 \times 0.9^2 = 32.4
$$

Since **72.9** is greater than **32.4**, the rover should prioritize going to position **1**, even though it is farther away.

### Policy ($ \pi $)

A **policy** ($ \pi $) defines the strategy of the rover: for each state, it dictates which action to take. Possible policies include:

1. **Greedy policy**: Always moves towards the highest reward state immediately.
2. **Exploratory policy**: Sometimes tries new actions to find better strategies.
3. **Discounted return policy**: Balances short-term and long-term rewards.

If the rover follows an **optimal policy**, it should compute the total expected reward for every possible action and pick the one that maximizes its long-term return.

## Markov Decision Process (MDP)

Reinforcement Learning problems are often modeled as **Markov Decision Processes (MDPs)**, which are defined by:

1. **Set of States (S)**: $ s_1, s_2, ..., s_n $
2. **Set of Actions (A)**: $ a_1, a_2, ..., a_m $
3. **Transition Probability (P)**: Probability of moving from one state to another given an action $ P(s' | s, a) $
4. **Reward Function (R)**: Defines the reward received when moving from $ s $ to $ s' $
5. **Discount Factor ($ \gamma $)**: Determines the importance of future rewards.

In our **Mars Rover** example:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/reinforcement-learning-03.png" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

- **States (S)**: {1, 2, 3, 4, 5, 6}
- **Actions (A)**: {Left (-1), Right (+1)}
- **Transition Probabilities (P)**: Deterministic (e.g., if the rover moves right, it always reaches the next state)
- **Reward Function (R)**:
  - $ R(1) = 100 $, $ R(6) = 40 $, $ R(2,3,4,5) = 0 $
- **Discount Factor ($ \gamma $)**: $ 0.9 $ (assumed)

<br/>

## State-Action Value Function ($Q(s,a)$)

The **State-Action Value Function**, denoted as $Q(s,a)$, represents the **expected return** when starting from state $s$, taking action $a$, and then following a policy $\pi$. Formally:

$$
Q(s,a) = \mathbb{E} \big[ G_t \mid S_t = s, A_t = a \big]
$$

This function helps the agent determine which action will lead to the highest reward in a given state.

<br/>

**Applying to Mars Rover**

Using our Mars rover example, we can estimate $Q(s,a)$ values for each state-action pair. Suppose:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/reinforcement-learning-04.png" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

- $Q(4, \text{left}) = 25$
- $Q(4, \text{right}) = 20$
- $Q(5, \text{right}) = 40$
- $Q(3, \text{left}) = 50$

The rover should always select the action with the highest $Q$ value to maximize rewards.

---

## Bellman Equation

The **Bellman Equation** provides a recursive relationship for computing value functions in reinforcement learning. It expresses the value of a state in terms of the values of successor states.

<br/>

**Understanding the Bellman Equation**

In reinforcement learning, an agent makes decisions in a way that maximizes future rewards. However, since future rewards are uncertain, we need a way to estimate them efficiently. The Bellman equation helps us do this by breaking down the value of a state into two components:

1. **Immediate Reward ($R(s,a)$)**: The reward received by taking action $a$ in state $s$.
2. **Future Rewards ($V(s')$)**: The expected value of the next state $s'$, weighted by the probability of reaching that state.

The Bellman equation is written as:

$$
V(s) = \max_a \Big[ R(s,a) + \gamma \sum_{s'} P(s' | s,a) V(s') \Big]
$$

where:

- $V(s)$: The value of state $s$.
- $R(s,a)$: The immediate reward for taking action $a$ in state $s$.
- $\gamma$: The discount factor ($0 \leq \gamma \leq 1$), which determines how much future rewards are considered.
- $P(s' | s,a)$: The probability of reaching state $s'$ after taking action $a$.
- $V(s')$: The value of the next state $s'$.

<br/>

**Example Calculation for Mars Rover**

Let’s assume:

- Moving from `4` to `3` has a reward of `-1`.
- Moving from `4` to `5` has a reward of `-1`.
- Position `1` has a reward of `100`.

For $s=4$:

$$
V(4) = \max \big[ -1 + \gamma V(3), -1 + \gamma V(5) \big]
$$

If we assume $V(3) = 50$ and $V(5) = 30$, and a discount factor $\gamma = 0.9$, we compute:

$$
V(4) = \max \big[ -1 + 0.9 \times 50, -1 + 0.9 \times 30 \big]
$$

$$
V(4) = \max \big[ -1 + 45, -1 + 27 \big]
$$

$$
V(4) = \max [44, 26] = 44
$$

Thus, the optimal value for state `4` is `44`, meaning the agent should prefer moving left toward `3`.

**Intuition Behind the Bellman Equation**

1. The Bellman equation decomposes the **value of a state** into its **immediate reward** and the **expected future reward**.
2. It allows us to compute values iteratively: we start with rough estimates and refine them over time.
3. It helps in **policy evaluation**—determining how good a given policy is.
4. It forms the foundation for **Dynamic Programming methods** like **Value Iteration** and **Policy Iteration**.

---

## Stochastic Environment (Randomness in RL)

In real-world applications, environments are often **stochastic**, meaning actions do not always lead to the same outcome.

**Stochasticity in the Mars Rover Example**

Suppose the Mars rover’s motors sometimes malfunction, causing it to move in the opposite direction with a small probability (e.g., 10% of the time). Now, the transition dynamics include:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/reinforcement-learning-05.png" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

- $P(s' = 5 | s = 4, a = \text{right}) = 0.9$
- $P(s' = 3 | s = 4, a = \text{right}) = 0.1$

This randomness makes decision-making more challenging. Instead of just considering rewards, the rover must now account for **expected rewards** and the probability of ending up in different states.

<br/>

**Impact on Decision-Making**

With stochastic environments, deterministic policies (always taking the best action) may not be optimal. Instead, an **exploration-exploitation** balance is needed:

- **Exploitation:** Following the best-known action based on past experience.
- **Exploration:** Trying new actions to discover potentially better rewards.

This concept is central to algorithms like **Q-Learning** and **Policy Gradient Methods**, which we will discuss in future sections.

---
