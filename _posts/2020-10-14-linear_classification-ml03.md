---
layout: post
data: 2020-10-14
tags: Machine-Learning
giscus_comments: True
title: Machine Learning - 03 Linear Classification
description: Include perceptron, Fisher's linear discriminant, logistic regression, Gaussian discriminant analysis and naive Bayes classifier. 
---

*The notes are based on the [session]( https://github.com/shuhuai007/Machine-Learning-Session). For the fundamental of linear algebra, one can always refer to [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/) and [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) for more details. Many thanks to these great works.*

* what?
{:toc}
# 0. Introduction

*The following introduction is derived from the [paper](http://pages.stat.wisc.edu/~wahba/stat860public/pdf1/liu.zhang.wu.lum.11.pdf).*

As a supervised learning technique, the goal of classification is to construct a classification rule based on a training set where both data and class labels are given. Once obtained, the classification rule can then be used for class prediction of new objects whose covariates are available.

Among various classification methods, there are two main groups: *soft* and *hard* classification. In particular, a soft classification rule generally estimates the class *conditional probabilities* explicitly and then makes the class prediction *based on the estimated probability*. Depending on whether calculating the conditional probability directly or approximating it by a model, there are *generative classifiers* and *discriminant classifiers* among the *soft* methods. In contrast, hard classification bypasses the requirement of class probability estimation and directly estimates the *classification boundary*. 

Typical soft classifiers include some traditional distribution-based likelihood approaches such as logistic regression. On the other hand, some margin-based approaches such as perceptron and the SVM, generally distributional assumption-free, belong to the class of hard classification methods.

We assume the data set is linearly separable in the following subsections.

# 1. Perceptron

Perceptron is a *hard* method for *binary classification*. Suppose we have i.i.d. data $$\mathcal{D}=\{(x_1,y_1), (x_2,y_2),\dots,(x_N,y_N)\},X=\{x_1,x_2,\dots,x_N\}, Y=\{y_1,y_2,\dots,y_N\}$$ where $$x_i\in\mathbb{R}^{d\times 1}$$ can be viewed as the feature and $$y_i\in\{-1,1\}$$ is the corresponding label. In particular, we denote $$X_{c1}=\{x_i\vert y_i=+1\}$$ and $$X_{c2}=\{x_i\vert y_i=-1\}$$ as the set of class $$c_1$$ and class $$c_2$$, respectively. Moreover, let $$N_1=\| X_{c1}\|$$ and $$N_2=\|X_{c2}\|$$, where $$N_1+N_2=N$$. The model of perceptron follows

$$f(w)=\text{sign}(w^Tx),$$

where $$w\in\mathbb{R}^{d\times 1}$$ and $$\text{sign}(\cdot)$$ is the sign function. Perceptron is actually an error-driven method. Specifically, for data $$(x_i,y_i)$$, the correctness of perceptron can be described as

$$y_iw^Tx_i\ge0\iff\begin{cases}w^Tx_i\ge0,\ f(w)=1,&\text{if } y_i=+1\\w^Tx_i<0,\ f(w)=-1, &\text{if }y_i=-1\end{cases}$$

Define $$\tilde D=\{(x,y)\vert y_iw^T x_i<0, i=1,\dots,N\}$$ be the set of data that was classified incorrectly. Then the loss function of the model can be defined as the size of $$\tilde D$$:

$$\mathcal{L}(w)=\sum_{i=1}^NI(y_iw^Tx_i<0),$$

where $$I(\cdot)$$ is the indicator function. Though such a loss function is intuitive, it is uncontinuous and can be hard to be optimized. From the standpoint of the model, to make $$y_iw^Tx_i\ge0$$ is equivalent to make $$y_iw^Tx_i$$ as larger as possible, thus we can transform the loss function into

$$\mathcal{L}(w)=\sum_{i=1}^N-y_iw^Tx_i,$$

which can be minimized by various optimization methods such as *stochastic gradient descent*.

# 2. Linear Discriminant Analysis

Now we introduce *linear discriminant analysis* (LDA), which is a method for *binary classification*. Note that in some materials, LDA is defined as a dimensionality reduction technique. Further, we introduce LDA method  in this note from a hard classification perspective. The soft perspective can be also found in other materials. The notations for data in this subsection are the same as that in subsection 1. We further define the mean

$$\bar x_{c1}=\frac{1}{N_1}\sum_{x\in X_{c1}}x,\quad \bar x_{c2}=\frac{1}{N_2}\sum_{x\in X_{c2}}x,$$

and the variance

$$S_{c1}=\frac{1}{N_1}\sum_{x\in X_{c1}}(x-\bar x_{c1})(x-\bar x_{c1})^T,\quad S_{c2}=\frac{1}{N_2}\sum_{x\in X_{c2}}(x-\bar x_{c2})(x-\bar x_{c2})^T.$$

> The idea of LDA is proposed by Ronald Fisher in 1988: maximize the distance between the mean of each class and minimize the spreading within the class itself.
>
> [hmm]: Thus, we come up with two measures: the within-class and the between-class. However, this formulation is only possible if we assume that the dataset has a Normal distribution. This assumption might bring a disadvantage because if the distribution of your data is significantly non-Gaussian, the LDA might not perform very well.

In LDA, we consider the ‘*projection*' of $$x$$:

$$z=w^Tx,$$

where $$w\in\mathbb{R}^{d\times 1}$$ is a unit vector to be learned. Specifically, the scalar $$z$$ is the length of the projection of $$x$$ on $$w$$, thus we can view such $$z$$ as the projection of $$x$$ into a *one dimensional subspace*. Note that the definition here is different from the definition of projection in *Introduction to Linear Algebra*.

Then we have the following definitions about the *mean*,

$$\bar z =\frac{1}{N}\sum_{i=1}^Nw^Tx_i,\quad \bar z_1=\frac{1}{N_1}\sum_{x\in X_{c1}}w^Tx,\quad \bar z_2=\frac{1}{N_2}\sum_{x\in X_{c2}}w^Tx.$$ 

Similarly, we have the definitions related to the variance as

$$S=\frac{1}{N}\sum_{i=1}^{N}(w^Tx_i-\bar z)^2,\quad S_1=\frac{1}{N_1}\sum_{x\in X_{c1}}(w^Tx-\bar z_1)^2,\quad S_2=\frac{1}{N_2}\sum_{x\in X_{c2}}(w^Tx-\bar z_2)^2.$$

Then we use the mean to define the distance between the two class and the variance to represent the spreading within the class itself. LDA is then to find the unit vector $$\hat w$$ that maximizes

$$\mathcal{J}(w)=\frac{(\bar z_1-\bar z_2)^2}{S_1+S_2}.$$

For the numerator, it follows that

$$\begin{aligned}(\bar z_1-\bar z_2)^2&=\left(\frac{1}{N_1}\sum_{x\in X_{c1}}w^Tx_i-\frac{1}{N_2}\sum_{x\in X_{c2}}w^Tx_i\right)^2\\&=w^T\left(\bar x_{c1}-\bar x_{c_2}\right)\left(\bar x_{c1}-\bar x_{c_2}\right)^Tw\end{aligned}.$$

For the denominator, it follows that

$$\begin{aligned}S_1+S_2&=\frac{1}{N_1}\sum_{x\in X_{c1}}(w^Tx-\bar z_1)^2+\frac{1}{N_2}\sum_{x\in X_{c2}}(w^Tx-\bar z_2)^2\\&=w^T\left[\frac{1}{N_1}\sum_{x\in X_{c1}}(x-\bar x_{c1})(x-\bar x_{c1})^T\right]w+w^T\left[\frac{1}{N_2}\sum_{x\in X_{c2}}(x-\bar x_{c2})(x-\bar x_{c2})^T\right]w\\&=w^TS_{c1}w+w^TS_{c2}w\\&=w^T(S_{c1}+S_{c2})w\end{aligned}.$$

Therefore, we have

$$\mathcal{J}(w)=\frac{w^TS_bw}{w^TS_ww},$$

where $$S_b=\left(\bar x_{c1}-\bar x_{c_2}\right)\left(\bar x_{c1}-\bar x_{c_2}\right)^T$$ represents the distance *between-class*, $$S_w=S_{c1}+S_{c2}$$ represents the spreading *within-class*. Such transformation is actually for computing derivation. $$J(w)$$ can be maximized by taking the derivative w.r.t $$w$$ and setting it to be $$0$$. Specifically,

$$\begin{aligned}\frac{\partial \mathcal{J}(w)}{\partial w}&=\frac{\left(\frac{\partial}{\partial w}w^TS_b w\right)w^TS_ww-w^TS_bw\left(\frac{\partial}{\partial w}w^TS_w w\right)}{(w^TS_ww)^2}\\&=\frac{(2S_bw)w^TS_ww-w^TS_bw(2S_ww)}{(w^TS_ww)^2}\end{aligned}.$$

Setting it to be 0 is equivalent to

$$\begin{aligned}(2S_bw)w^TS_ww-w^TS_bw(2S_ww)&=0\\ (w^TS_bw)S_ww&=S_bw(w^TS_ww)\\S_w w&=\frac{w^TS_ww}{w^TS_bw}S_bw\end{aligned}.$$

As $$w\in\mathbb{R}^{d\times 1}$$ and $$S_w,S_b\in\mathbb{R}^{d\times d}$$, the term $$(w^TS_ww)/(w^TS_bw)\in\mathbb{R}$$. For convenience, we denote it as $$\lambda$$. Then we have an equivalent *generalized eigenvalue problem*

$$S_ww=\lambda S_bw.$$

If one of $$S_b$$ and $$S_w$$ has full rank, the generalized eigenvalue problem can be converted into a standard eigenvalue problem. However, to solve the problem entails complex computation. We now assume $$S_w^{-1}$$ exists. Recall that $$w$$ is a unit vector. Thus what we need to care is only the direction of $$w$$:

$$\begin{aligned}\hat w&\propto \lambda S_w^{-1}S_bw\\&\propto \lambda S_w^{-1}\left(\bar x_{c1}-\bar x_{c_2}\right)\left(\bar x_{c1}-\bar x_{c_2}\right)^Tw\\&\propto\lambda_1S_w^{-1}\left(\bar x_{c1}-\bar x_{c_2}\right)\\&\propto S_w^{-1}\left(\bar x_{c1}-\bar x_{c_2}\right)\end{aligned},$$

where $$\lambda_1=\lambda \left(\bar x_{c1}-\bar x_{c_2}\right)^Tw$$ is a scalar as $$\left(\bar x_{c1}-\bar x_{c_2}\right)^Tw\in\mathbb{R}$$.

# 3. Discriminant Classifiers

Discriminant classifiers focus on the classification problem directly. Specifically, discriminant classifiers  model the posterior $$P(Y\vert X)$$, then makes the class prediction based on the estimated probability.

## 3.1. Logistic Regression

Logistic regression inputs the result of a *linear regression* to a *sigmoid function* to make classification. A sigmoid function is

$$\sigma(z)=\frac{1}{1+e^{-z}}$$

which maps $$z\in(-\infty,+\infty)$$ into a probability $$[0,1]$$. Therefore we can model the posterior probability as

$$P(y=1|x;w)=\sigma(w^Tx)=\frac{1}{1+e^{-w^Tx}},$$

where $$w$$ is the parameter to be learned. As for $$P(y=-1\vert x;w)$$, it can be obtained by $$1-P(y=1\vert x;w)$$ since we are considering a binary classification problem. However, for a supervised learning technique, it would be convenient to consider both two labels into one function. To this end, we set the label value to be $$y_i\in\{0,1\}$$ . Moreover, we denote $$P(y_i=1\vert x_i;w)$$ and $$P(y_i=0\vert x_i;w)$$ as $$p_{i\cdot 1}$$ and $$p_{i\cdot 0}$$, respectively. Then logistic regression is to find $$\hat w$$ that maximizes

$$\mathcal{J}(w)=P(Y|X;w)=\prod_{i=1}^N p_{i\cdot 1}^{y_i}p_{i\cdot 0}^{1-y_i}.$$

Given the dataset $$\mathcal{D}$$, such maximization problem can be solved by *maximum likelihood estimation*:

$$\begin{aligned}\hat w&=\arg\max_w \log P(Y\vert X;w)\\&=\arg\max_w \log\prod_{i=1}^N p_{i\cdot 1}^{y_i}p_{i\cdot 0}^{1-y_i}\\&=\arg\max_w \sum_{i=1}^N(y_i\log p_{i\cdot 1}+(1-y_i)\log p_{i\cdot 0})\\&=\arg\max_w \sum_{i=1}^N[y_i\log \sigma(w^Tx_i)+(1-y_i)\log (1-\sigma(w^Tx_i))]\end{aligned}.$$

> The term $$\sum_{i=1}^N(y_i\log \sigma(w^Tx_i)+(1-y_i)\log (1-\sigma(w^Tx_i)))$$ is actually the negative of *cross entropy* over $$P(Y)$$ and $$\sigma(w^TX)$$.

To solve the above problem, one can refer to SGD method.

# 4. Generative Classifiers

For a binary classification problem, we actually have no need to know the specfic value of $$P(y=1\vert x)$$ and $$P(y=0\vert x)$$. What matters is whether $$P(y=1\vert x)>P(y=0\vert x)$$ or not. Unlike discriminant methods which model and compute the posterior probability directly, in *generative classifiers*, we compare the posterior probability in an indirect way. Specifically, by *Bayes's theorem*, we have

$$P(Y|X)=\frac{P(X|Y)P(Y)}{P(X)}\propto P(X|Y)P(Y).$$

Therefore, to compare the posterior probability is to compare the union probability. The classification predicted by generative classifiers is

$$\hat y=\arg\max_{y\in\{0,1\}}P(y\vert x)=\arg\max_{y\in\{0,1\}}P(x\vert y)P(y).$$

In generative classifier methods, a key problem is how to model the likelihood $$P(x\vert y)$$ and the prior $$P(y)$$. 

## 4.1. Naive Bayes Classifier

Naive Bayes classifier is the simplest generative classifier. For a binary classification problem, suppose the feature of $$x_i$$ is composed of $$(x_{i1},x_{i2},\dots,x_{id})$$. Then naive Bayes classifier assumes not only the independence among the data but also that *every pair of the feature is independent*, *i.e.,*

$$x_{im}\vert y_i\perp x_{in}\vert y_i,m,n=1,2,\dots,d \text{ and }m\ne n.$$

Then the likelihood becomes

$$P(x_i\vert y_i)=\prod_{j=1}^dP(x_{ij}\vert y_i).$$

Further, it models the prior and each feature as,

$$y_i\sim\text{Bern}(\phi),\quad x_{ij}\vert y_i\sim\mathcal{N}(\mu_{j},\sigma_j^2),$$

where $$\phi,\mu_j$$, and $$\sigma_{j}$$ are parameters that can be learned by MLE method. Note that such model is just a common case. The key idea of naive Bayes classifier is its independence assumption. Specifically, naive Bayes classifier is not a single method but a family of methods. By assuming the independence, it can be extremely fast compared with other classification methods.

## 4.2. Gaussian Discriminant Analysis

As a generative method, *Gaussian discriminant analysis* (GDA) models the prior and the likelihood as follows,

$$y\sim\text{Bern}(\phi),\quad x\vert y=0\sim\mathcal{N}(\mu_1,\Sigma),\quad x\vert y=1\sim\mathcal{N}(\mu_2,\Sigma),$$

where $$\phi,\mu_1,\mu_2$$, and $$\Sigma$$ are parameters to be learned. We define $$w=(\phi, \mu_1,\mu_2,\Sigma)$$. Then GDA is to find $$\hat w$$ that maximizes

$$\begin{aligned}\mathcal{J}(w)&=\log \prod_{i=1}^N P(x_i\vert y_i;\mu_1,\mu_2,\Sigma)P(y_i;\phi)\\&=\sum_{i=1}^N\left(\log P(x_i\vert y_i;\mu_1,\mu_2,\Sigma)+\log P(y_i;\phi)\right)\end{aligned}.$$

Similar to the case in section 3.1, we represent the likelihood and the prior as

$$P(x_i\vert y_i;\mu_1,\mu_2,\Sigma)=\rho^{y_i}(\mu_1,\Sigma)\rho^{1-y_i}(\mu_2,\Sigma),\quad P(y_i;\phi)=\phi^{y_i}(1-\phi)^{1-y_i},$$

where $$\rho(\mu_1,\Sigma)$$ and $$\rho(\mu_2,\Sigma)$$ are the PDF of $$x\vert y=0$$ and $$x\vert y=1$$, respectively. Then it follows that

$$\begin{aligned}\mathcal{J}(w)&=\sum_{i=1}^N\left(\log\rho^{y_i}(\mu_1,\Sigma)\rho^{1-y_i}(\mu_2,\Sigma)+\log\phi^{y_i}(1-\phi)^{1-y_i}\right)\\&=\sum_{i=1}^N\left(y_i\log \rho(\mu_1,\Sigma)+(1-y_i)\log\rho(\mu_2,\Sigma)+y_i\log\phi+(1-y_i)\log(1-\phi)\right)\end{aligned}.$$

Then to find $$\hat w$$ that maximizes $$\mathcal{J}(w)$$ is equivalent to set the derivation of $$\mathcal{J}(w)$$ w.r.t $$w$$ to be zero.

For $$\hat \phi$$:

$$\begin{aligned}\frac{\partial \mathcal{J}(w)}{\partial\phi}=\sum_{i=1}^N\left(\frac{y_i}{\phi}-\frac{1-y_i}{1-\phi}\right)\end{aligned}.$$

Solving $$\partial \mathcal{J}(w)/\partial\phi=0$$, we have

$$\hat \phi=\frac{1}{N}\sum_{i=1}^N y_i=\frac{N_1}{N}.$$

For $$\hat\mu_1$$ (or $$\hat\mu_2$$ likewise):

$$\begin{aligned}\frac{\partial\mathcal{J}(w)}{\partial\mu_1}&=\frac{\partial\left(\sum_{i=1}^N y_i\log \rho(\mu_1,\Sigma)\right)}{\partial\mu_1}\\&=\frac{\partial\left(\sum_{i=1}^Ny_i\log\left(\frac{1}{(2\pi)^{d/2}\vert\Sigma\vert^{1/2}}\exp\left(-\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)\right)\right)\right)}{\partial\mu_1}\\&=\frac{\partial\left(-\frac{1}{2}\sum_{i=1}^Ny_i(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)\right)}{\partial\mu_1}\\&=\frac{\partial\left(-\frac{1}{2}\sum_{i=1}^Ny_i(x_i^T\Sigma^{-1}x_i-x_i^T\Sigma^{-1}\mu_1-\mu_1^T\Sigma^{-1}x_i+\mu_1^T\Sigma^{-1}\mu_1)\right)}{\partial\mu_1}\\&=\sum_{i=1}^Ny_i\left(\Sigma^{-1}\mu_1-\Sigma^{-1}x_i\right)\end{aligned}.$$

> [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf):
>
> $$\frac{\partial x^Ta}{\partial x}=\frac{\partial a^Tx}{\partial x}=a.$$
>
> $$\frac{\partial x^TBx}{\partial x}=(B+B^T)x.$$

Solving $$\partial \mathcal{J}(w)/\partial\mu_1=0$$, we have

$$\hat \mu_1=\frac{\sum_{i=1}^N y_ix_i}{\sum_{i=1}^N y_i}=\bar x_{c1}.$$

Similarly, we have

$$\hat\mu_2=\bar x_{c2}.$$

For $$\hat\Sigma$$, we first consider the following transformation

$$\begin{aligned}\sum_{i=1}^N\left(y_i\log \rho(\mu_1,\Sigma)+(1-y_i)\log\rho(\mu_2,\Sigma)\right)&=\sum_{x\in X_{c1}}\log\rho(\mu_1,\Sigma)+\sum_{x\in X_{c2}}\log\rho(\mu_2,\Sigma)\end{aligned}.$$

As $$\Sigma$$ is shared by both $$\rho(\mu_1,\Sigma)$$ and $$\rho(\mu_2,\Sigma)$$, we consider the expansion of $$\rho(\mu_1,\Sigma)$$ for example,

$$\begin{aligned}\sum_{x\in X_{c1}}\log\rho(\mu_1,\Sigma)&=\sum_{x\in X_{c1}}\log\left(\frac{1}{(2\pi)^{d/2}\vert\Sigma\vert^{1/2}}\exp\left(-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\right)\right)\\&=\underbrace{\sum_{x\in X_{c1}}-\frac{d}{2}\log2\pi}_{\text{constant }\lambda_1}-\sum_{x\in X_{c1}}\left(\frac{1}{2}\log\vert \Sigma\vert+\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\right)\\&=\lambda_1-\frac{N_1}{2}\log\vert\Sigma\vert-\frac{1}{2}\sum_{x\in X_{c1}}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\end{aligned}.$$

For the third term, notice that $$(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\in\mathbb{R}$$, thus

$$\begin{aligned}\sum_{x\in X_{c1}}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)&=\text{tr}\left(\sum_{x\in X_{c1}}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\right)\\&=\text{tr}\left(\sum_{x\in X_{c1}}(x-\mu_1)(x-\mu_1)^T\Sigma^{-1}\right)\\&=N_1\text{tr}\left(S_{c1}\Sigma^{-1}\right)\end{aligned}.$$

> [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf):
>
> $$\text{tr}(ABC)=\text{tr}(CAB)=\text{tr}(BCA).$$
>
> $$\frac{\partial\text{tr}(AB)}{\partial A}=B^T.$$
>
> $$\frac{\partial \vert A\vert}{\partial A}=\vert A\vert A^{-1}.$$

Hence,

$$\sum_{x\in X_{c1}}\log\rho(\mu_1,\Sigma)=\lambda_1-\frac{N_1}{2}\log\vert\Sigma\vert-\frac{N_1}{2}\text{tr}\left(S_{c1}\Sigma^{-1}\right).$$

Similarly,

$$\sum_{x\in X_{c2}}\log\rho(\mu_2,\Sigma)=\lambda_2-\frac{N_2}{2}\log\vert\Sigma\vert-\frac{N_2}{2}\text{tr}\left(S_{c2}\Sigma^{-1}\right).$$

Then it follows that

$$\begin{aligned}\frac{\partial\mathcal{J}(w)}{\partial\Sigma}&=\frac{\partial \left(\sum_{i=1}^N\left(y_i\log \rho+(1-y_i)\log\rho\right)\right)}{\partial\Sigma}\\&=\frac{\partial \left(\sum_{x\in X_{c1}}\log\rho+\sum_{x\in X_{c2}}\log\rho\right)}{\partial\Sigma}\\&=-\frac{1}{2}\frac{\partial \left(N\log\vert\Sigma\vert+N_1\text{tr}\left(S_{c1}\Sigma^{-1}\right)+N_2\text{tr}\left(S_{c2}\Sigma^{-1}\right)\right)}{\partial\Sigma}\\&=-\frac{1}{2}\left(N\Sigma^{-1}-N_1S_{c1}\Sigma^{-2}-N_2S_{c2}\Sigma^{-2}\right)\end{aligned}.$$

By setting $$\partial \mathcal{J}(w)/\partial\Sigma=0$$, we finally arrive at

$$\hat\Sigma=\frac{N_1S_{c1}+N_2S_{c2}}{N}.$$

# 5. Conclusion

In this post, we introduced five linear classifiers. Among these models, $$\mathcal{L}(w)$$ is to be minimized, while $$\mathcal{J}(w)$$ is to be maximized. We omitted the prediction part of a classification problem. What we focused is actually how to model these data, especially in those generative cases.

This post is obviously a long story. Moreover, there are many other things stoped my writing occasionally these days. It definitely has some logical problems, let alone typos, to be fixed. Anyway, I made it. Hope next time I can do better. 



