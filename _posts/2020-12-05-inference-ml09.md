---
layout: post
data: 2020-12-05
tags: Machine-Learning
giscus_comments: True
title: Machine Learning - 09 Exact Inference of Graphical Models
description: A very abstract post as it involves graphs and trees while providing no figures and examples.
---

*The notes are based on the [session]( https://github.com/shuhuai007/Machine-Learning-Session), [CS228-notes](https://ermongroup.github.io/cs228-notes/) and [PRML](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf). For the fundamental of probability, one can refer to [Introduction to Probability](https://drive.google.com/file/d/1VmkAAGOYCTORq1wxSQqy255qLJjTNvBI/view). Many thanks to these great works.*

* what?
{:toc}
# 0. Introduction

In the last [post](https://2ez4ai.github.io/2020/11/29/probabilistic_graphical_models-ml08/), we introduce graphical models which is capable of representing random variables and the conditional independences among them. We now consider the problem of inference in graphical models. Particularly, we wish to compute posterior distributions of one or more nodes conditioned on some other known (observed) nodes, and the techniques we shall talk in this post are for *exact inference*.

# 1. Variable Elimination

We first consider the marginal inference in a *chain* Bayesian network which has the following joint distribution,

$$P(x_1,x_2,\dots,x_n)=P(x_1)\prod_{i=2}^nP(x_i\vert x_{i-1}).$$

Suppose we want to infer the marginal distribution $P(x_n)$. A naive way to do that is marginalizing $x_1,x_2,\dots,x_{n-1}$:

$$P(x_n)=\sum_{x_1}\sum_{x_2}\dots\sum_{x_{n-1}}P(x_1,x_2,\dots,x_{n-1}).$$

However, as there are $n-1$ variables each with $k$ states, the computation needs to sum the probability over $k^{n-1}$ values and would scale exponentially with the length of the chain. To simplify the computation, we can leverage *variable elimination*.

The key of *variable elimination* is in *rearranging the order* of the summations and the multiplications. Specifically, the marginal distribution follows that

$$\begin{aligned}P(x_n)&=\sum_{x_1}\sum_{x_2}\dots\sum_{x_{n-1}}P(x_1)\prod_{i=2}^nP(x_i\vert x_{i-1})\\&=\sum_{x_{n-1}}P(x_n\vert x_{n-1})\cdot \sum_{x_{n-2}}P(x_{n-1}\vert x_{n-2})\cdot\dots\cdot \sum_{x_{1}}P(x_{2}\vert x_{1})P(x_1).\end{aligned}$$

Such a rearrangement works because multiplication is distributive over addition,

$$ab+ac=a(b+c),$$

where the number of arithmetic operations are reduced from three (the left-hand side) to two (the right-hand side). Before we move on, we generalize the new expression to a Markov network since every Bayesian network can be transformed into a Markov network. For the chain Bayesian network we considered, we can remove all the arrows of the graph to obtain a Markov network. The obtained Markov network would have maximum cliques $$\{x_1,x_2\}$$, …, $$\{x_{n-2},x_{n-1}\}$$ and $$\{x_{n-1},x_{n}\}$$ and the corresponding potentials are

$$\begin{aligned}\psi_{1,2}(x_1,x_2)&=P(x_2\vert x_1)P(x_1),\\&\vdots\\\psi_{x_{n-2},x_{n-1}}(x_{n-2},x_{n-1})&=P(x_{n-2}\vert x_{n-1}),\\\psi_{n-1,n}(x_{n-1},x_n)&=P(x_{n-1}\vert x_n).\end{aligned}$$

The rearrangement for the Markov network then follows that

$$P(x_n)=\sum_{x_{n-1}}\psi_{n-1,n}(x_{n-1}, x_{n})\cdot \sum_{x_{n-2}}\psi_{n-2,n-1}(x_{n-2}, x_{n-1})\cdot\dots\cdot \sum_{x_{1}}\psi_{1,2}(x_1,x_2).$$

Now the computation composes of $n-1$ summations. More importantly, unlike the previous one sums over $k^{n-1}$ values, this expression allows each term only need to sum over $k\times k$ values. Specifically, the sum

$$\sum_{x_i}\psi_{x_i,x_{i+1}}(x_{i},x_{i+1})$$

only involves two variables and thus the summation is over $k\times k$ values. Then the overall computation is of $O(nk^2)$ complexity, which is much better than the naive $O(k^n)$ method.

However, the variable elimination (VE) method requires an ordering over the variables. In fact, the running time of VE on different orderings would vary greatly, while to find the best ordering is still an NP-hard problem. Moreover, VE method for $P(x_n)$ can be hard to be generalized to other marginal distribution as it does not store the intermediate results.

# 2. Belief Propagation

For convenience, we consider undirected graphs with tree structure, where the optimal variable elimination ordering for node $x_i$ is the post-order iteration of the subtree rooted at $x_i$. The relationship between any two directly connected nodes is decided by which node the tree is rooted at and how far the two nodes are away from the root: the close one is the parent of the farther one.

## 2.1. Message

For a tree graph, its maximum cliques contains only two nodes. By VE algorithm, to compute the marginal $P(x_i)$, we need to eliminate all nodes that are in the subtree of $x_i$. For node $x_j$, the elimination involves computing $\sum_{x_j}\psi_{x_j,x_k}(x_j,x_k)m_{j,k}$ where $x_k$ is the parent of $x_j$ in the tree. The term $m_{j,k}$ can be thought of a *message* that $x_j$ sends to $x_k$ about the subtree rooted at $x_j$. Similarly, the computing result can be viewed as

$$m_{k,l}=\sum_{x_j}\psi_{x_j,x_k}(x_j,x_k)m_{j,k}$$

that contains the information for $x_l$, the parent of $x_k$, about the subtree rooted at $x_k$. By doing so, at the end of VE, $x_i$ would receive messages from all of its immediate children and then marginalize them out to yield the final marginal.

Suppose that after computing $P(x_i)$, we are interested in computing $P(x_k)$ as well. If we use VE algorithm again, we can find that the computation also involves the messages $m_{j,k}$ as node $x_k$ is still the parent of node $x_j$. Moreover, such a message is exactly the same as the one used in computing $P(x_i)$ since the graph structure does not change. Therefore, it is easy to find that if we store the intermediary messages of VE, we can obtain other marginals quickly.

## 2.2. Sum-Product

Belief propagation can be viewed as a combination of VE and *caching*. For each edge between $x_i$ and $x_j$, the messages passing on it are $m_{i,j}$ and $m_{j,i}$, which depends on the marginal we want to determine. After computing all these messages, one can compute any marginals with these messages.

Belief propagation:

- Set a node, for example, node $x_i$, as the root;
- For each $x_j$ in $N(x_i)$, *i.e.,* the neighborhood of $x_i$, collect the messages sent to $x_i$:

$$m_{j,i}=\sum_{x_j}\psi_{x_i,x_j}(x_i,x_j)\psi_{x_j}(x_j)\prod_{k\in N(x_j)\setminus i}m_{k,j};$$

- For each $x_j$ in $N(x_i)$, collect the messages sent from $x_i$:

$$m_{i,j}=\sum_{x_i}\psi_{x_j,x_i}(x_j,x_i)\psi_{x_i}(x_i)\prod_{k\in N(x_i)\setminus j}m_{k,i}.$$

By doing so, we can obtain all the messages with $$2\vert E\vert$$ steps, where $E$ is the set of edges. Then for any marginal we have

$$P(x_i)=\psi_i(x_i)\prod_{k\in N(x_i)}m_{k,i}.$$

## 2.3 Max-Product

We now consider a problem of finding the set of values that have the largest probability so that

$$\hat{\text{x}}=\arg\max_{\text{x}} P(\text{x}).$$

Notice that

$$\max_{\text{x}} P(\text{x})=\max_{\text{x}_1}\dots \max_{\text{x}_n}P(\text{x}).$$

By *sum-product*, we have

$$\begin{aligned}\max_{\text{x}} P(\text{x})&=\max_{x_1}\dots \max_{x_n}\psi_i(x_i)\prod_{k\in N(x_i)}m_{k,i}\\&=\max_{x_n}\max_{x_n-1{}}\psi_{x_n,x_{n-1}}(x_n,x_{n-1})\max_{x_{n-2}}\psi_{x_{n-1},x_{n-2}}(x_{n-1},x_{n-2})\\&\quad\ \dots\max_{x_1}\psi_{x_2,x_1}(x_2,x_1)\psi_{x_1}(x_1).\end{aligned}$$

Such a method for maximizing *max-product* is known as *max-product* algorithm.

# 3. Conclusion

In this post, we briefly introduced two algorithms for *exact inference* in graphical models. Given a proper order of nodes, *variable elimination* algorithm is efficient. However, the finding of the proper order is an NP-hard problem. Besides, each query of marginals needs running the algorithm, during which the computation can be highly redundant. To improve computing efficiency, *belief propagation* stores the intermediate results as messages. After that, one can get any marginal by the messages. Moreover, we can also exploit those messages to determine the values of random variables with the largest probability, which is known as *max-product*.

