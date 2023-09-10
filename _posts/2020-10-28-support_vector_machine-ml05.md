---
layout: post
data: 2020-10-28
tags: Machine-Learning
giscus_comments: True
title: Machine Learning - 05 Support Vector Machine
description: Include support vector machine and Lagrange duality. 
---

*The notes are based on the [session]( https://github.com/shuhuai007/Machine-Learning-Session). For the fundamental of linear algebra, one can always refer to [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/) and [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) for more details. Many thanks to these great works.*

* what?
{:toc}
# 0. Introduction

Support vector machine (SVM) is a supervised learning method for classification and regression analysis. It is one of the most robust prediction method. Here we mainly consider its applications in classification. Specifically, for the data of $$d$$-dimensional, we want to know whether we can separate classes with a $$(d-1)$$-dimensional *hyperplane*. In particular, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class. According to whether the dataset is linearly separable or not, there are *hard-margin* SVM, *soft-margin* SVM and *kernel* SVM.

# 1. Hard-margin SVM

Hard-margin SVM works only when data is completely linearly separable without any errors.

 

<div align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png" width="350" />
</div>
## 1.1. Problem Formulation

Suppose we have data set $$\mathcal{D}=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$$ where $$x_i\in\mathbb{R}^{d\times 1}$$ is the data feature and $$y_i\in\{-1,1\}$$ is the corresponding class label. A case where $$d=2$$ is shown in the figure above. The hyperplane that separates the data is defined as

$$w^Tx-b=0,$$

where $$w\in\mathbb{R}^{d\times 1}$$ and $$b\in\mathbb{R}$$ are parameters to be learned. Then like what we arrived in [perceptron](https://2ez4ai.github.io/2020/10/14/linear_classification-ml03/), a correct classifier should ensure that

$$y_i(w^Tx_i-b)>0,\forall i=1,2,\dots,N.$$

We further define the *margin* as the parallel lines that has the minimum distance from the data to the hyperplane. In *hard-margin* SVM, we need to find the *maximum-margin* hyperplane that maximizes the distance, which can be described by

$$\begin{aligned}\max_{w,b}\min_{i}&\quad \frac{\vert w^Tx_i-b\vert}{\vert\vert w\vert\vert}\\\text{s.t.}&\quad y_i(w^Tx_i-b)>0,\forall i=1,2,\dots,N\end{aligned}.$$

The problem further is equivalent to

$$\begin{alignat*}{3}\max_{w,b}\min_{i}&\quad \frac{y_i(w^Tx_i-b)}{\vert\vert w\vert\vert}\implies\max_{w,b}\frac{1}{\vert\vert w\vert\vert}\min_i&\quad y_i(w^Tx_i-b)\\\text{s.t.}&\quad y_i(w^Tx_i-b)>0,\forall i=1,2,\dots,N\end{alignat*}.$$

For the constraint $$y_i(w^Tx_i-b)>0, \forall i=1,2,\dots,N$$, there exists a positive parameter $$r>0$$ such that

$$\min_i y_i(w^Tx_i-b)=r.$$

As there are many $$w,b$$ available for the separation as long as they have the same directions. We add a new constraint that $$r=1$$. Then it follows that

$$\min_i y_i(w^Tx_i-b)=y_i((w_{\text{old}}^T/r)x_i-b_{\text{old}}/r)=1.$$

The problem then is transformed into

$$\begin{alignat*}{3}&&\max_{w,b}\frac{1}{\vert\vert w\vert\vert}\min_i y_i(w^Tx_i-b)&\implies\min_{w,b}\quad \frac{1}{2}w^Tw\\&&\text{s.t.}\quad y_i(w^Tx_i-b)>0&\implies\text{s.t.}\quad y_i(w^Tx_i-b)\ge 1,i=1,2,\dots,N\end{alignat*},$$

which is a linearly constrained *quadratic optimization* (QP) problem.

## 1.2. Lagrange Duality

The following content is about [convex optimization](https://web.stanford.edu/~boyd/cvxbook/). In section 1, we have the *primal problem*

$$\begin{aligned}\min_{w,b}&\quad \frac{1}{2}w^Tw\\\text{s.t.}&\quad y_i(w^Tx_i-b)\ge 1,i=1,2,\dots,N\end{aligned}.$$

The *Lagrangian* for the problem is a function defined as

$$L(w,b,\lambda)=\frac{1}{2}w^Tw+\sum_{i=1}^N\lambda_i\left(1-y_i\left(w^Tx_i-b\right)\right),$$

where $$\lambda_i\ge 0$$ is the *Lagrange multiplier* associated with the constraints. Consider the problem of $$\max_\lambda L(w,b,\lambda)$$,

$$\max_{\lambda\ge 0} L(w,b,\lambda)=\begin{cases}\frac{1}{2}w^Tw+\infty&\text{if }\exists i\in\{1,2,\dots,N\}\text{ s.t. }1-y_i(w^Tx_i-b)>0,\\\frac{1}{2}w^Tw+0&\text{otherwise.}\end{cases}$$

The problem makes sense (non infinity) only when the original constraint is satisfied. In that case, the primal problem is equivalent to

$$\begin{aligned}\min_{w,b}\max_\lambda&\quad L(w,b,\lambda)\\\text{s.t.}&\quad \lambda_i\ge 0,i=1,2,\dots,N\end{aligned}.$$

Then we define the *Lagrange dual function* for the primal problem as

$$g(\lambda)=\min_{w,b}L(w,b,\lambda).$$

> Actually, the correct definition of the *Lagrange dual function* should be
>
> $$g(\lambda)=\inf_{w,b}L(w,b,\lambda).$$
>
> Here we assume the minimum exists and the infimum is the minimum for understanding.

The *Lagrange dual problem* of the original problem is then defined as

$$\begin{aligned}\max_{\lambda}&\quad g(\lambda)\\\text{s.t.}&\quad \lambda_i\ge 0,i=1,2,\dots,N\end{aligned}.$$

The dual problem is introduced for its convexity. Specifically, notice that the infimum (minimum in this case) of $$g(\lambda)$$ is unconstrained as opposed to the original constrained minimization problem. Further, $$g(\lambda)$$ is concave with respect to $$\lambda$$ regardless of the original problem.

|                        Primal Problem                        |                    Lagrange Dual Problem                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| $$\begin{aligned}\min_{w,b}\max_\lambda&\quad L(w,b,\lambda)\\\text{s.t.}&\quad \lambda_i\ge 0,i=1,2,\dots,N\end{aligned}$$ | $$\begin{aligned}\max_{\lambda}\min_{w,b}&\quad L(w,b,\lambda)\\\text{s.t.}&\quad \lambda_i\ge 0,i=1,2,\dots,N\end{aligned}$$ |

A natural problem is whether the two problems are equivalent. Obviously, the equivalence is obtained if and only if

$$\min_{w,b}\max_{\lambda} L(w,b,\lambda)=\max_{\lambda}\min_{w,b} L(w,b,\lambda).$$

If the equation holds, we say the *strong duality* holds. It can also be shown that the *weak duality* always holds as

$$\min_{w,b}\max_{\lambda} L(w,b,\lambda)\ge\max_{\lambda}\min_{w,b} L(w,b,\lambda).$$

*Proof:*

Obviously, we have

$$\max_\lambda L(w,b,\lambda)\ge L(w,b,\lambda)\ge\min_{w,b}L(w,b,\lambda).$$

Define $$F(w,b)=\max_\lambda L(w,b,\lambda)$$ and $$G(\lambda)=\min_{w,b}L(w,b,\lambda)$$. According the above inequality, it follows that

$$F(w,b)\ge G(\lambda)\implies\min_{w,b}F(w,b)\ge\max_\lambda G(\lambda).$$

Therefore we have

$$\min_{w,b}\max_{\lambda} L(w,b,\lambda)\ge\max_{\lambda}\min_{w,b} L(w,b,\lambda).\tag*{$$\blacksquare$$}$$



Solving the dual problem in fact is used to find nontrivial lower bounds for difficult original problems. In our case, the strong duality holds for the linearly constrained QP problem. Thus to solve the primal problem is to solve the dual problem.

## 1.3. Karush–Kuhn–Tucker Conditions

For the primal problem and its dual problem, if the strong duality holds, then *Karush–Kuhn–Tucker (KKT) conditions* are satisfied as

- (a). Primal Feasibility:

  $$y_i(w^Tx_i-b)\ge1,i=1,2,\dots,N$$

- (b). Dual Feasibility:

  $$\lambda_i\ge 0,i=1,2,\dots,N$$

- (c). Complementary Slackness:

  $$\hat\lambda_i\left(1-y_i\left(\hat{w}^Tx_i-\hat b\right)\right)=0,i=1,2,\dots,N$$

- (d). Zero gradient of Lagrangian with respect to $$w,b$$:

  $$\frac{\partial L}{\partial b}=0,\quad \frac{\partial L}{\partial w}=0.$$

The conditions (a) and (b) are the original constraints. As for condition (c), recall that we define $$y_i(w^Tx_i-b)=1$$ for the data that is exactly $$1/\vert\vert w\vert\vert$$ away from the hyperplane $$w^Tx-b=0$$, *i.e.,* on the margin of the hyperplane. For those which are not on the margin, to satisfy KKT conditions, it must follow that

$$\hat\lambda_k=0,k\in\{i\vert y_i(w^Tx_i-b)>1,i=1,2,\dots,N\}.$$

The condition (d) is for the dual problem. Specifically, we consider the unconstrained problem $$\min_{w,b}L(w,b,\lambda)$$ in the dual problem. For the differentiable function $$L(w,b,\lambda)$$ , by *Fermat’s theorem*, the extremum exists when condition (d) is satisfied, $$i.e.,$$

$$\frac{\partial L}{\partial b}=\sum_{i=1}^N\lambda_iy_i=0,\\\frac{\partial L}{\partial w}=w-\sum_{i=1}^N\lambda_iy_ix_i=0.$$

Solving the equations we have

$$\sum_{i=1}^N\lambda_iy_i=0,\forall b,\quad \hat w=\sum_{i=1}^N\lambda_iy_ix_i.$$

Plugging them into $$L(w,b,\lambda)$$, we can transform the problem into

$$\begin{aligned}\max_{\lambda\ge0}&\quad \min_{w,b}L(w,b,\lambda)\\\implies\max_{\lambda\ge0}&\quad \frac{1}{2}\sum_{i=1}^{N}\left(\lambda_iy_ix_i^T\right)\sum_{i=1}^{N}\left(\lambda_iy_ix_i\right)+\sum_{i=1}^N\lambda_i-\sum_{i=1}^N\lambda_iy_i\left(\sum_{j=1}^N\lambda_jy_jx_j^T\right)x_i\\\text{s.t.}&\quad \sum_{i=1}^N\lambda_iy_i=0\\\implies\max_{\lambda\ge0}&\quad -\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^N\left(\lambda_i\lambda_jy_iy_jx_i^Tx_j\right)+\sum_{i=1}^N\lambda_i\\\text{s.t.}&\quad \sum_{i=1}^N\lambda_iy_i=0\\\implies \min_\lambda&\quad \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^N\left(\lambda_i\lambda_jy_iy_jx_i^Tx_j\right)-\sum_{i=1}^N\lambda_i\\\text{s.t.}&\quad \lambda_i\ge0,i=1,2,\dots,N\\&\quad \sum_{i=1}^N\lambda_iy_i=0\end{aligned}$$

The optimal $$\hat\lambda$$ can be obtained by *sequential minimal optimization* (SMO) algorithm. Here we assume we already have the optimal value. Notice we have $$\sum_{i=1}^N\hat\lambda_iy_i=0$$, which means there exists at least one $$\hat\lambda_k\ne0$$ otherwise $$\hat w=0$$. According our analysis, $$\hat\lambda_k\ne 0$$ only when

$$y_k(w^Tx_k-b)-1=0.$$

Therefore we have the solution

$$\hat w=\sum_{i=1}^N\hat\lambda_iy_ix_i,$$

$$\hat b=\sum_{i=1}^N\hat\lambda_iy_ix_i^Tx_k-y_k.$$

Accordingly, the hyperplane is the linear combination of the data on the margin with corresponding $$\hat\lambda_k>0$$. We call those data *support vectors* from where the name SVM comes.

# 2. Soft-margin SVM

In practice, there are noise and outliers among the data, which makes the data nonlinearly separable. In that case, hard-margin fails to work. Now we introduce *soft-margin SVM* which extends SVM to the nonlinearly separable data. Recall that in hard-margin SVM, we have the constraint

$$y_i(w^Tx_i-b)\ge 1,i=1,2,\dots,N$$

which confines the model to the linearly separable case. To extent the model to general cases, we introduce *loss function*, which can be defined as

- The number of wrongly classifying:

  $$\text{loss}=\sum_{i=1}^N I(y_i(w^Tx_i-b)<1),$$

  where $$I(\cdot)$$ is the indicator function. However, such loss function is not differentiable.

- The sum of the distances between the hyperplane and the outliers:

  $$\begin{aligned}\text{loss}_i&=\begin{cases}0&y_i(w^Tx_i-b)\ge 1\\1-y_i(w^Tx_i-b)&y_i(w^Tx_i-b)<1\text{ (wrongly classified)}\end{cases}\\&=\max\{0, 1-y_i(w^Tx_i-b)\}.\\\text{loss}&=\sum_{i=1}^N\text{loss}_i,\end{aligned}$$

  which is called **hinge loss**.

However, the $$\max$$ in the hinge loss is not differentiable neither. We now adapt the original constraint as

$$y_i(w^Tx_i-b)\ge 1-\xi_i,i=1,2,\dots,N$$

where $$\xi_i\ge0$$ and $$\sum_{i=1}^N\xi_i\le$$ constant are called *slack variables*. The slack variables is introduced to allow for some points to be on the wrong side of the margin. Specifically, for the points that are on the wrong side, it will break the original constraint $$y_i(w^Tx_i-b)\ge 1$$ as

$$y_i(w^Tx_i-b)=\xi< 1.$$

With slack variables, such classification is allowed as long as

$$\xi\ge 1-\xi_i\implies\xi_i\ge1-\xi.$$

Moreover, we do not want the $$\xi_i$$ to be too large to distinguish points correctly. Thus we have the new formulation

$$\begin{aligned}\min_{w,b}&\quad \frac{1}{2}w^Tw+C\sum_{i=1}^N\xi_i\\\text{s.t.}&\quad y_i(w^Tx_i-b)\ge 1-\xi_i\\&\quad \xi_i\ge 0\\&\quad i=1,2,\dots,N\end{aligned},$$

where $$C$$ is the *cost* parameter that determines to what extent we allow for outliers. To solve the problem one can refer to the hard-margin case as they are actually similar.

# 3. Conclusion

In this post, we first introduced hard-margin SVM for linearly separable data. By introducing a loss function and slack variables, soft-margin SVM allows for noise and outliers so that it can handle non linear case. The two models can both be solved by *convex optimization* methods. For convex optimization, we briefly reviewed *Lagrange duality*, *Slater’s condition* and *KKT conditions*.