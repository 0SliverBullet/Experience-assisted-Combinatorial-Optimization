# Knapsack Problem

## 0. greedy algorithm

start with **greedy algorithm**, which is the quality baseline of feasible solutions

think where is the problem from

- select from the smaller weight
- select from the bigger value
- select from the bigger "value density"
- ...

## 1. a knapsack model

**The (1-Dimensional) Knapsack Problem** 

Given a set of items $\mathbb{I}$, each item $i \in \mathbb{I}$ characterized by its weight $w_i$ and its value $v_i$, a capacity $K$ for a knapsack, find the subset of items in $\mathbb{I}$ that has maximum value and does not exceed the capacity $K$ of the knapsack. 

**Optimization Models**

- choose some decision variables
- express the problem constraints in terms of these variables
- express the objection function 

**Problem Formulation**
$$
\begin{align}
\text{maximize} &\quad \sum_{i\in\mathbb{I}}v_ix_i \\
\text{subject to} & \quad \sum_{i\in\mathbb{I}}w_ix_i \leq K\\
&\quad x_i \in \{0,1\} \quad (i\in \mathbb{I})\\
\end{align}
$$

## 2. dynamic programming

**Basic Principle**

- divide and conquer
- bottom up computation (different from recursive method)



assume that $\mathbb{I}=\{1,2,...,n\}$, $O(k,j)$ denotes the optimal solution to the knapsack problem with capacity $k$ and items $[1, ..., j]$
$$
\begin{align}
\text{maximize} &\quad \sum_{i\in\{1,..,j\}}v_ix_i \\
\text{subject to} & \quad \sum_{i\in\{1,..,j\}}w_ix_i \leq k\\
&\quad x_i \in \{0,1\} \quad (i\in \{1,..,j\})\\
\end{align}
$$
We are interested in finding out the best value $O(K,n)$

**Recurrence Relations (Bellman Equations)**
$$
\begin{align}
O(k,j)= \begin{cases}
0 & \quad \text{if}\ j=0\\
\text{max}(O(k,j-1),v_j+O(k-w_j,j-1))& \quad \text{if}\ w_j \leq k \\
O(k,j-1) &\quad \text{otherwise}
\end{cases}
\end{align}
$$
**Time Complexity**

the algorithm is in fact **exponential** in terms of the input size

## 3. Branch and Bound

- Iterative two steps

– branching 

– bounding

- Branching

– split the problem into a number of subproblems

•like in exhaustive search

- Bounding

– find an **optimistic estimate** of the best solution to the subproblem: Relaxation!

• maximization: upper bound

• minimization: lower bound

## 4. Search Strategies

- Search Strategies

– depth-first, best-first, least-discrepancy

– many others

- Depth-first

– prunes when a node estimation is worse than the best found solution

– memory efficient

- Best-First

– select the node with the best estimation

- Least-Discrepancy

– trust a greedy heuristic 