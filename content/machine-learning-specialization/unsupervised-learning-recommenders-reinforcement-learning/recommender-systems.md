# Recommender Systems

<!-- toc -->

Recommender systems are everywhere in our digital lives, from Netflix suggesting movies based on our watch history to Amazon recommending products based on our previous purchases. These systems aim to predict what users might like based on their past behavior or the attributes of the items themselves.

# Collaborative Filtering

Collaborative filtering is one of the most widely used techniques in recommender systems. It works by leveraging the behavior and preferences of users to make predictions about what they might like. Instead of relying on the characteristics of items themselves, collaborative filtering focuses on the interactions between users and items.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/recommender-systems-01.webp" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

Imagine a streaming service like Netflix. If many users who watched "The Matrix" also watched "Inception," the system might recommend "Inception" to a user who has already watched "The Matrix." This works because the system assumes that similar users have similar tastes.

There are two main types of collaborative filtering:

1. **User-based Collaborative Filtering**: Recommendations are made by finding users with similar preferences.
2. **Item-based Collaborative Filtering**: Recommendations are made by finding similar items based on user interactions.

## User-based Collaborative Filtering

Consider a movie recommendation system with four users (A, B, C, D) and seven movies (M1, M2, M3, M4, M5, M6, M7). The users have rated some of the movies on a scale from 1 to 5, but not every user has watched every movie. Our goal is to predict which **unwatched movie** user **D** would like the most and recommend it.

Below is the ratings matrix:

| User | M1  | M2  | M3  | M4  | M5  | M6  | M7  |
| ---- | --- | --- | --- | --- | --- | --- | --- |
| A    | 5   | 3   | 4   | -   | 2   | -   | 1   |
| B    | 4   | -   | 5   | 3   | 1   | 2   | -   |
| C    | 3   | 5   | -   | 4   | -   | 1   | 2   |
| D    | -   | 4   | 5   | 2   | 1   | -   | -   |

User **D** has not rated **M1, M6, and M7**, so we need to predict which one they are most likely to enjoy.

### Finding Similar Users

We use a similarity measure to identify users most similar to **D**. A common choice is **cosine similarity**, defined as:

$$
\text{sim}(u, v) = \frac{ \sum_{i \in I} r_{ui} r_{vi} }{ \sqrt{ \sum_{i \in I} r_{ui}^2 } \sqrt{ \sum_{i \in I} r_{vi}^2 } }
$$

where:

- $ r\_{ui} $ is the rating of user $ u $ for item $ i $.
- $ I $ is the set of items rated by both users.

Computing similarity between **D** and other users:

Using **cosine similarity**, we compare D with other users:

| User | M2  | M3  | M5  |
| ---- | --- | --- | --- |
| A    | 3   | 4   | 2   |
| D    | 4   | 5   | 1   |

$$
sim(D, A) = \frac{(4 \times 3) + (5 \times 4) + (1 \times 2)}{\sqrt{(4^2 + 5^2 + 1^2)} \times \sqrt{(3^2 + 4^2 + 2^2)}} = 0.974
$$

Similarly, we compute:

| User | M3  | M4  | M5  |
| ---- | --- | --- | --- |
| B    | 5   | 3   | 1   |
| D    | 5   | 2   | 1   |

<br/>

| User | M2  | M4  |
| ---- | --- | --- |
| C    | 5   | 4   |
| D    | 4   | 2   |

$$
sim(D, B) = 0.988, \quad sim(D, C) = 0.979
$$

Since **B** is most similar to **D**, we estimate **D**'s ratings for the unwatched movies **(M1, M6, M7)** using a weighted average:

$$
\hat{r}_{D, j} = \bar{r}_D + \frac{ \sum_{u} \, 	ext{sim}(D, u) \cdot (r_{u, j} - \bar{r}_u) }{ \sum_{u} |	ext{sim}(D, u)| }
$$

#### Predicting Rating for M1

Using the weighted sum formula:

$$
\hat{r}_{D, M1} = \frac{(sim(D, A) \times r_{A, M1}) + (sim(D, B) \times r*{B, M1}) + (sim(D, C) \times r*{C, M1})}{sim(D, A) + sim(D, B) + sim(D, C)}
$$

<br/>

$$
\hat{r}\_{D, M1} = \frac{(0.974 \times 5) + (0.988 \times 4) + (0.979 \times 3)}{0.974 + 0.988 + 0.979} = 3.998
$$

Repeating for M6 and M7, we get:

$$
\hat{r}_{D, M6} = 1.494, \quad \hat{r}_{D, M7} = 1.505
$$

Since **M1 has the highest predicted rating (3.998)**, we recommend M1 to user D.

- **Predicted rating for M1: 3.998**
- **Predicted rating for M6: 1.494**
- **Predicted rating for M7: 1.505**

Since **M1** has the highest predicted rating, we recommend **M1** to **D**.

## Item-based Collaborative Filtering

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/recommender-systems-03.png" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

Rather than finding similar users, **item-based collaborative filtering** identifies similar items based on how users have rated them. The main idea is that if two movies are rated similarly by multiple users, they are likely to be similar.

### Finding Similar Items

To determine item similarity, we use cosine similarity but compute it between movie rating vectors instead of user rating vectors.

Computing similarity between **M1, M6, and M7** and other movies:

- **sim(M1, M3)** = 0.82
- **sim(M6, M2)** = 0.78
- **sim(M7, M5)** = 0.73

Since **M3** is most similar to **M1**, we predict **D's rating for M1** based on **D's rating for M3**:

$$
\hat{r}_{D, M1} = \frac{ \sum_{i} \, 	ext{sim}(M1, i) \cdot r_{D, i} }{ \sum_{i} |	ext{sim}(M1, i)| }
$$

After calculations:

- **Predicted rating for M1: 4.1**
- **Predicted rating for M6: 3.7**
- **Predicted rating for M7: 3.6**

Since **M1** has the highest predicted rating, we again recommend **M1** to **D**.

### Conclusion

- **User-based filtering** finds similar users and recommends based on their preferences.
- **Item-based filtering** finds similar items and predicts ratings based on a user's history.
- Both methods predicted that **D would like M1 the most**, making it the best recommendation.
- These techniques can be combined for **hybrid recommender systems** to improve accuracy.

# Content-Based Filtering

Content-based filtering recommends items to users by analyzing the characteristics of items a user has interacted with and comparing them with the characteristics of other items. Unlike collaborative filtering, which relies on user-item interactions, content-based filtering uses item metadata, such as genre, actors, or textual descriptions, to determine similarities.

## Understanding Content-Based Filtering

In content-based filtering, each item is represented by a set of features. Users are assumed to have a preference for items with similar features to those they have previously liked. The recommendation process typically involves:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/recommender-systems-02.png" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

1. **Feature Representation**: Representing items in terms of feature vectors.
2. **User Profile Construction**: Creating a preference model for each user based on past interactions.
3. **Similarity Computation**: Comparing new items with the user’s profile to generate recommendations.
4. **Generating Recommendations**: Ranking items based on similarity scores and recommending the top ones.

To better understand this approach, let’s consider an example.

## Example: Movie Recommendation

We have a dataset of seven movies, each described by three features: genre, director, and lead actor. Additionally, four users have rated some of these movies on a scale of 1 to 5.

### Movie Feature Representation

Each movie is represented using a feature vector based on genre, director, and actors. We assign numerical values to categorical features using one-hot encoding.

| Movie | Action | Comedy | Drama | Sci-Fi | Director A | Director B | Actor X | Actor Y |
| ----- | ------ | ------ | ----- | ------ | ---------- | ---------- | ------- | ------- |
| M1    | 1      | 0      | 0     | 1      | 1          | 0          | 1       | 0       |
| M2    | 0      | 1      | 1     | 0      | 0          | 1          | 0       | 1       |
| M3    | 1      | 1      | 0     | 0      | 1          | 0          | 1       | 0       |
| M4    | 0      | 0      | 1     | 1      | 0          | 1          | 0       | 1       |
| M5    | 1      | 0      | 1     | 0      | 1          | 0          | 1       | 0       |
| M6    | 0      | 1      | 0     | 1      | 0          | 1          | 0       | 1       |
| M7    | 1      | 0      | 1     | 0      | 1          | 0          | 1       | 0       |

### User Ratings

| User | M1  | M2  | M3  | M4  | M5  | M6  | M7  |
| ---- | --- | --- | --- | --- | --- | --- | --- |
| A    | 5   | 3   | 4   | -   | 2   | -   | 1   |
| B    | 4   | -   | 5   | 3   | 1   | 2   | -   |
| C    | 3   | 5   | -   | 4   | -   | 1   | 2   |
| D    | -   | 4   | 5   | 2   | 1   | -   | -   |

### Step 1: Constructing User Profiles

For each user, we compute a preference vector by averaging the feature vectors of the movies they have rated, weighted by their ratings.

For example, user D has rated three movies: M2 (4), M3 (5), and M4 (2). Their profile vector is computed as:

$$ P*D = \frac{4 \times V*{M2} + 5 \times V*{M3} + 2 \times V*{M4}}{4 + 5 + 2} $$

This results in a vector representing user D’s preferences.

### Step 2: Computing Similarity Scores

To recommend a new movie (e.g., M6 or M7), we compute the cosine similarity between the user’s preference vector and the feature vector of the candidate movies:

$$ \text{sim}(P*D, V*{Mi}) = \frac{P*D \cdot V*{Mi}}{||P*D|| \times ||V*{Mi}||} $$

Where $ P*D \cdot V*{Mi} $ is the dot product and $ ||P*D|| $ and $ ||V*{Mi}|| $ are the magnitudes.

### Step 3: Generating Recommendations

By ranking the movies based on their similarity scores with the user’s profile, we can recommend the highest-ranked movie. If M6 has a similarity of 0.85 and M7 has 0.75, we recommend M6.

## Advantages and Challenges of Content-Based Filtering

### Advantages:

- Personalized recommendations based on individual preferences.
- Does not suffer from the cold start problem for items.
- No need for extensive user interaction data.

### Challenges:

- Requires well-defined item features.
- Struggles with the cold start problem for new users.
- Limited to recommending items similar to those already interacted with.

By integrating deep learning techniques, such as word embeddings and neural networks, content-based filtering can improve accuracy and extend recommendations beyond direct similarities.

## Principal Components Analysis

### Reducing the Number of Features

PCA helps in dimensionality reduction, keeping only the most significant features.

### PCA Algorithm

Given a dataset $ X $:

1. Compute the mean and subtract it from $ X $.
2. Compute the covariance matrix $ \Sigma $.
3. Compute eigenvalues and eigenvectors of $ \Sigma $.
4. Select the top $ k $ eigenvectors.
5. Project $ X $ onto the new feature space.

### PCA in Code

```python
from sklearn.decomposition import PCA
import numpy as np

X = np.random.rand(100, 50)  # 100 samples, 50 features
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)
```

PCA helps reduce computational cost while preserving essential information for recommendations.

#### Real-Life Example: Face Recognition

Facebook uses PCA to compress images while retaining key facial features for recognition.

---
