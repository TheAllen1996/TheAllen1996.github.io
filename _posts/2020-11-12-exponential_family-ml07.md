---
layout: post
data: 2020-11-12
tags: Machine-Learning
giscus_comments: True
title: Machine Learning - 07 Exponential Family
description: Introduce concepts of exponential family.
---

*The notes are based on the [session]( https://github.com/shuhuai007/Machine-Learning-Session) and the [material](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf). For the fundamental of linear algebra, one can always refer to [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/) and [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) for more details. Many thanks to these great works.*

* what?
{:toc}
# 0. Introduction

An exponential family is a family of distributions which share some properties in common. The real message of this note is the simplicity and elegance of the exponential family. Once the ideas are mastered, it is often easier to work within the general exponential family framework than with specific instances.

# 1. Exponential Family

Given one real-vector parameter $$\mathbf{\theta}=[\theta_1,\theta_2,\dots,\theta_d]^T$$, we define an *exponential family* of probability distributions as those distributions whose density have the following general form:

$$f_X(x\vert\eta)=h(x)\exp\left(\eta^T T(x)-A(\eta)\right).$$

**Canonical parameter**: $\eta$ is called *canonical*, or *natural parameter* (function), which can be viewed as a transformation of $\mathcal{\theta}$. The set of values of $\eta$ is always convex.

**Sufficient statistic**: Given a data set sampled from $$f_X(x\vert\eta)$$, the sufficient statistic $T(x)$ is a function of the data that holds all information the data set provides with regard to the unknown parameter $\mathbf{\theta}$.

**Log-partition function**: $A(\eta)$ is the *log-partition function* to normalize $$f_X(x\vert \eta)$$ to be a probability distribution,

$$A(\eta)=\log\left(\int_{X}h(x)\exp(\eta^T T(x))\text{d}x\right).$$

In the following sections, we will discuss them detailedly.

# 2. Sufficient Statistic

Consider the problem of estimating the unknown parameters by *maximum likelihood estimation* (MLE) in exponential family cases. Specifically, for an *i.i.d.* data set $$\mathcal{D}=\{x_1,x_2,\dots,x_N\}$$, we have the log likelihood

$$\mathcal{L}(\eta\vert\mathcal{D})=\log\left(\prod_{i=1}^Nh(x_i)\right)+\eta^T\left(\sum_{i=1}^NT(x_i)\right)-NA(\eta).$$

By *MLE*, we have the estimation $\hat\eta$ when its gradient with respect to $\eta$ is zero:

$$\mathcal{L}’(\eta\vert\mathcal{D})=\sum_{i=1}^NT(x_i)-NA’(\eta)=0.$$

Solving the equation, we have

$$A’(\hat\eta)=\frac{1}{N}\sum_{i=1}^NT(x_i),$$

which is the general formula of MLE for the parameters in the exponential family. Further, notice that our formula involves the data only via the sufficient statistic $T(x_i)$. This gives the operational meaning to *sufficiency*—for the purpose of estimating parameters we retain only the sufficient statistic.

#  3. Log-partition Function

As we mentioned in section 1, $A(\eta)$ can be viewed as a normalization factor. In fact, $A(\eta)$ is not a degree of freedom in the specification of an exponential family density; it is determined once $T(x)$ and $h(x)$ are determined. The relation between $A(\eta)$ and $T(x)$ can be further characterized by

$$\begin{aligned}A’(\eta)&=\frac{\text{d}\log\left(\int_{X}h(x)\exp(\eta^T T(x))\text{d}x\right)}{\text{d}\eta}\\&=\frac{\int_{X}h(x)\exp(\eta^T T(x))\cdot  T(x)\text{d}x}{\int_{X}h(x)\exp(\eta^T T(x))\text{d}x}\\&=\frac{\int_{X}h(x)\exp(\eta^T T(x))\cdot T(x)\text{d}x}{\exp({A(\mathbf{\theta})})}\\&=\int_{X}\underbrace{h(x)\exp(\eta^T T(x)-A(\eta))}_{f_X(x\vert \eta)}\cdot T(x)\text{d}x\\&=\mathbb{E}_{f_X(x\vert\eta)}[T(x)].\end{aligned}$$

Further, we have

$$\begin{aligned}A’’(\eta)&=\int_{X}f_X(x\vert \eta)\cdot(T(x)-A’(\eta)) T(x)\text{d}x\\&=\int_{X}f_X(x\vert \eta)\cdot(T(x))^2\text{d}x-A’(\eta)\int_{X}f_X(x\vert \mathbf{\eta})\cdot T(x)\text{d}x\\&=\mathbb{E}_{f_X(x\vert\eta)}[(T(x))^2]-\left(\mathbb{E}_{f_X(x\vert\eta)}[T(x)]\right)^2\\&=var[T(x)],\end{aligned}$$

which also shows that $A(\eta)$ is convex as $var[T(x)]\ge 0$.

# 4. Maximum Entropy

The entropy of $P$ with distribution $p(x)$ supported on $X$ is 

$$H(P)=\mathbb{E}_{P}[-\log p(x)].$$

The *maximum entropy* principle is that: given some constraints (prior information) about the distribution $P$, we consider all probability distributions satisfying said constraints such that the constraints are being utilized as *objective* as possible, *i.e.,* be as uncertain as possible.

For example, consider the case where the very constraint is $\sum_Xp(x)=1$, which formulates

$$\begin{aligned}\max&\quad H(P)\\\text{s.t.}&\quad \sum_{X}p(x)=1.\end{aligned}$$

By the definition we have

$$H(P)=-\sum_{i=1}^{\vert X\vert}p(x_i)\log p(x_i).$$

Then the *Lagrangian* for the optimization problem is

$$L(P,\lambda)=\sum_{i=1}^{\vert X\vert}p(x_i)\log p(x_i)+\lambda\left(1-\sum_{i=1}^{\vert X\vert}p(x_i)\right).$$

Setting the first derivation of the Lagrangian to be zero yields

$$\frac{\partial L}{\partial p(x_i)}=0\implies \hat{p}(x_i)=\exp(\lambda-1),$$

which gives that

$$\hat{p}(x_1)=\hat{p}(x_2)=\dots=\hat{p}(x_{\vert X\vert})=\frac{1}{\vert X\vert},$$

*i.e.,* the distribution with maximum entropy is *uniform distribution*.

We now consider a general case where $p(x)$ is continuous with a general constraint $\mathbb{E}_P[\Phi(x)]=\alpha$, where $\Phi(x)=[\phi_1(x),\phi_2(x),\dots,\phi_d(x)]\in\mathbb{R}^d$ and $\alpha=[\alpha_1,\alpha_2,\dots,\alpha_d]\in\mathbb{R}^d$, which formulates

$$\begin{aligned}\max&\quad H(P)\\\text{s.t.}&\quad \mathbb{E}_P[\Phi(x)]=\alpha\\\implies\min&\quad \int_Xp(x)\log p(x)\text{d}x\\\text{s.t.}&\quad \int_X p(x)\phi_i(x)\text{d}x=\alpha_i,\ i=1,2,\dots, d,\\&\quad \int_X p(x)\text{d}x=1.\end{aligned}$$

Similarly, we obtain the Lagrangian as

$$L(P,\theta,\lambda)=\int_X p(x)\log p(x)\text{d}x+\sum_{i=1}^d\theta_i\left(\alpha_i-\int_X p(x)\phi_i(x)\text{d}x\right)+\lambda\left(\int_X p(x)\text{d}x-1\right).$$

By treating the density $P=[p(x)]_{x\in X}$ as a finite vector such that $\int_X p(x)\text{d}x$ is similar to $\sum_X p(x)$, we have

$$\begin{aligned}\frac{\partial L}{\partial p(x)}&=\frac{\partial }{\partial p(x)}\left(\sum_X p(x)\log p(x)-\sum_{i=1}^d\theta_i\sum_X p(x)\phi_i(x)+\lambda\sum_X p(x)\right)\\&=1+\log p(x)-\sum_{i=1}^d\theta_i\phi_i(x)+\lambda\\&=1+\log p(x)-\theta^T\Phi(x)+\lambda.\end{aligned}$$

Setting the derivation to be zero for all $x$, we have

$$p(x)=\exp\left\{\theta^T\Phi(x)-(\lambda+1)\right\},$$

which is in the exponential family form with

$$\begin{aligned}\eta&=\theta,\\T(x)&=\Phi(x),\\A(\eta)&=\lambda+1,\\h(x)&=1.\end{aligned}$$

# 5. Gaussian Distribution

In this section, we consider an example, Gaussian distribution, which is of the exponential family and exemplifies the properties we mentioned above. 

We first rewritten the PDF of one-dimension Gaussian distribution to show it is in the exponential family .

*Proof:* Given unknown parameter $$\mathbf{\theta}=[\mu,\sigma^2]$$, the Gaussian density can be written as follows,

$$\begin{aligned}f_X(x\vert \mathbf{\theta})&=\frac{1}{\sqrt{2\pi}\sigma}\exp\left\{-\frac{1}{2\sigma^2}(x-\mu)^2\right\}\\&=\frac{1}{\sqrt{2\pi}}\exp\left\{\frac{\mu}{\sigma^2}x-\frac{1}{2\sigma^2}x^2-\frac{1}{2\sigma^2}\mu^2-\log\sigma\right\}\\&=\frac{1}{\sqrt{2\pi}}\exp\left\{\begin{bmatrix}\frac{\mu}{\sigma^2}&-\frac{1}{2}\sigma^2\end{bmatrix}\begin{bmatrix}x\\x^2\end{bmatrix}-\left(\frac{\mu^2}{2\sigma^2}+\log\sigma\right)\right\},\end{aligned}$$

which is in the exponential family form with

$$\begin{aligned}\eta&=\begin{bmatrix}\frac{\mu}{\sigma^2}&-\frac{1}{2\sigma^2}\end{bmatrix}^T,\\T(x)&=\begin{bmatrix}x&x^2\end{bmatrix}^T,\\A(\eta)&=\frac{\mu^2}{2\sigma^2}+\log\sigma=-\frac{\eta_1^2}{4\eta_2}-\frac{1}{2}\log(-2\eta_2),\\h(x)&=\frac{1}{\sqrt{2\pi}}.\end{aligned}$$

$$\tag*{$\blacksquare$}$$

Then we verify the relation between the sufficient statistic and MLE method.

*Proof:* Given a data set $$\mathcal{D}=\{x_1,x_2,\dots,x_N\}$$, as we mentioned in section 2, we can derive the parameters via the sufficient statistic as follows,

$$\begin{cases}A’(\eta)=\frac{1}{N}\sum_{i=1}^NT(x)\implies\begin{cases}A’(\hat\eta_1)=\frac{1}{N}\sum_{i=1}^N x_i\\A’(\hat\eta_2)=\frac{1}{N}\sum_{i=1}^Nx_i^2\end{cases}\\A(\eta)=-\frac{\eta_1^2}{4\eta_2}-\frac{1}{2}\log(-2\eta_2)\implies\begin{cases}A’(\hat\eta_1)=-\frac{\eta_1}{2\eta_2}=\hat\mu\\A’(\hat\eta_2)=\frac{\eta_1^2}{4\eta_2^2}-\frac{1}{2\eta_2}=\hat\sigma^2+\hat\mu^2\end{cases}\end{cases}$$

Solving the equations we have

$$\hat\mu=\frac{1}{N}\sum_{i=1}^Nx_i,\quad \hat\sigma=\frac{1}{N}\sum_{i=1}^Nx_i^2-\hat\mu^2,$$

which is consistent with the result in the [post](https://2ez4ai.github.io/2020/09/28/intro-ml01/). $\tag*{$\blacksquare$}$

Now we show that

$$A’’(\hat\eta)=var[T(x)].$$

*Proof*: Firstly, we have

$$\begin{cases}A’’(\hat\eta_1)=-\frac{1}{2\eta_2}=\sigma^2\\A’’(\hat\eta_2)=-\frac{\eta_1^2}{2\eta_2^3}+\frac{1}{2\eta_2^2}=4\sigma^2\mu^2+2\sigma^4\end{cases}$$

For $$T(x)=\begin{bmatrix}x&x^2\end{bmatrix}^T$$, we have

$$var[x]=\sigma^2, \text{ as }x\sim\mathcal{N}(\mu,\sigma^2),$$

and

$$var[x^2]=\mathbb{E}[x^4]-\left(\mathbb{E}[x^2]\right)^2.$$

For $\mathbb{E}[x^2]$, it follows that

$$\mathbb{E}[x^2]=var[x]+(\mathbb{E}[x])^2=\sigma^2+\mu^2.$$

For $\mathbb{E}[x^4]$, to compute it we leverage *moment generating functions* which follows that

$$M_X(t)=e^{\mu t+\frac{1}{2}\sigma^2t^2},\quad \mathbb{E}[x^4]=M^{(4)}_X(0).$$

After a laborious computing, we have

$$\begin{aligned}var[x^2]&=\mathbb{E}[x^4]-\left(\mathbb{E}[x^2]\right)^2\\&=3\sigma^4+6\sigma^2\mu^2+\mu^4-\sigma^4-2\sigma^2-\mu^4\\&=4\sigma^2\mu^2+2\sigma^4.\end{aligned}$$

Therefore, we have

$$A’’(\hat\eta)=\begin{bmatrix}var[x]\\var[x^2]\end{bmatrix}.\tag*{$\blacksquare$}$$

Finally, we show that $X\sim\mathcal{N}(\mu,\sigma^2)$ is the distribution that maximizes the entropy over all distributions $P$ satisfying

$$\mathbb{E}_P\left[\left(\frac{X-\mu}{\sigma}\right)^2\right]=1.$$

*Proof:* Consider the expression we formulated in section 4,

$$p(x)=\exp\left\{\theta^T\Phi(x)-(\lambda+1)\right\},$$

which maximizes the entropy while satisfying $\mathbb{E}_P[\Phi(x)]=\alpha$. Now letting

$$\begin{aligned}\alpha&=1,\\\Phi(x)&=\frac{(x-\mu)^2}{\sigma^2},\\\theta&=-\frac{1}{2},\\\exp\{-\lambda-1\}&=\frac{1}{\sqrt{2\pi}\sigma}.\end{aligned}$$

Therefore we have

$$p(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left\{-\frac{1}{2\sigma^2}(x-\mu)^2\right\}.\tag*{$\blacksquare$}$$



# 6. Conclusion

In this post, we briefly introduced the basic form of the exponential family. Then we discussed its properties from three perspectives: sufficient statistic, log-partition function and maximum entropy. Moreover, with one-dimension Gaussian distribution, we exemplified the properties.

