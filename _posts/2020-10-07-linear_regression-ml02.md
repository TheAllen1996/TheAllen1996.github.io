---
layout: post
data: 2020-10-07
tags: Machine-Learning
giscus_comments: True
title: Machine Learning - 02 Linear Regression
description: A naive model of machine learning. Introduce three perspectives to the least squares method.
---

*The notes are based on the [session]( https://github.com/shuhuai007/Machine-Learning-Session). Many thanks to the great work.*

* what?
{:toc}
# 1. Least Squares Method

Suppose we have the IID data $$\mathcal{D}=\{(x_1,y_1), (x_2,y_2),\dots,(x_N,y_N)\},$$ where $x_i\in\mathbb{R}^{d\times 1}, y_i\in\mathbb{R}, i=1,2,\dots,N$. One can view $x_i$ as a feature vector and $y_i$ is the corresponding label. Denote

$$X=(x_1,x_2,\dots,x_N)^T=\begin{pmatrix}x_{11}&x_{12}&\dots&x_{1d}\\x_{21}&x_{22}&\dots&x_{2d}\\\vdots&\vdots&\ddots&\vdots\\x_{N1}&x_{12}&\dots&x_{1d}\end{pmatrix}_{N\times d},\quad Y=\begin{pmatrix}y_1\\y_2\\\vdots\\y_N\end{pmatrix}_{N\times 1}.$$

The problem is given the data set, we need to find a function to fit these data while minimizing the sum of squared error. In this case, we only focus on linear function:

$$f(w)=w^Tx,$$

where $w\in\mathbb{R}^{d\times 1}$ is the unknown weight we need to learn. Here we ignore the bias term as it can be represented by adding a new dimension to the variables. The *loss function* of the least squares method is defined as

$$\mathcal{L}(w)=\sum_{i=1}^N||w^Tx_i-y_i||^2.$$

In this case, $w^Tx_i\in\mathbb{R}$ and $y_i\in\mathbb{R}$. Thus we can expand the loss function as

$$\begin{aligned}\mathcal{L}(w)&=\underbrace{\begin{pmatrix}w^Tx_1-y_1&w^Tx_2-y_2&\dots&w^Tx_N-y_N\end{pmatrix}}_{w^T\begin{pmatrix}x_1&x_2&\dots&x_N\end{pmatrix}-\begin{pmatrix}y_1&y_2&\dots&y_N\end{pmatrix}}\begin{pmatrix}w^Tx_1-y_1\\w^Tx_2-y_2\\\vdots\\w^Tx_N-y_N\end{pmatrix}\\&=(w^TX^T-Y^T)(Xw-Y)\\&=w^TX^TXw-2w^TX^TY+Y^TY\end{aligned}.$$

Such expansion is for the derivative of the loss. Thus we have the optimal weight where

$$\begin{alignat*}{3}\hat w&=\arg\min_w \mathcal{L}(w)\Rightarrow\frac{\partial \mathcal{L}(w)}{\partial w}&=0\end{alignat*}.$$

Solving the equation, we have

$$\hat{w}=(X^TX)^{-1}X^TY.$$

# 2. Least Squares Method - A Matrix Perspective

Knowledge about [vectors and space](http://math.mit.edu/~gs/linearalgebra/) is required in this section. We consider using a new approximation linear function

$$h(w)=Xw,$$

where $w\in\mathbb{R}^{d\times 1}$ and it is quite similar to $f(w)$ we defined in section 1. One should keep in mind that a matrix multiple a vector yields a linear combination of the **column vectors** of the matrix. We will use $X_1,X_2,\dots,X_N$ to represent the column vector of $X$. Ideally, we want to find the $w$ subject to

$$X w=Y.$$

If such $w$ exists, then we can solve it directly and easily. However, in practice error is inevitable and such $w$ often does not exist, which means $Y$ is not a linear combination of $X_i$ and they do not share the same space. What we can do is finding a linear combination of $X_i$ so that it has the least Euclidean distance to $Y$. 

As I am too lazy to depict the picture, I would like to give an imaginable example for that. One can also get the picture easily with these examples. The following discussion is based on a three-dimensional space with $xyz$ axes.

**Ideal case**: Suppose we have the data $$x_1, x_2, x_3 = \{(1,0,0)^T,(0,1,0)^T,(0,0,1)^T\}, Y=(1,1,1)^T$$. Thus

$$X=(x_1,x_2, x_3)^T=\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix},\quad Y=\begin{pmatrix}1\\1\\1\end{pmatrix}.$$

With the knowledge of vectors, it can be easily found that we can obtain $Y$ by $1\cdot X_1+1\cdot X_2+1\cdot X_3$, which gives the solution for $X\hat{w}=Y$ as $\hat{w}=(1,1,1)^T$. 

**Practical case**: In practice, things may be different. Suppose we have $x_1, x_2, x_3 = \\{(1,0.2,0.5)^T,(0,1,0)^T,(0,0,0)^T\\}, Y=(1,1,1)^T$. Thus

$$X=(x_1,x_2, x_3)^T=\begin{pmatrix}1&0.2&0.5\\0&1&0\\0&0&0\end{pmatrix},\quad Y=\begin{pmatrix}1\\1\\1\end{pmatrix}.$$

With a manual drafting, one can find there is no way to get $Y$ by the linear combination of $X_1,X_2$ and $X_3$: all the $X_i$ lies in $xy$ surface while $Y$ exists in $xyz$ space. In such case, what we can do is finding a line on $xy$ surface as close to $Y$ as possible. Obviously, the closest one is the projection of $Y$ on $xy$ surface, which is $(1,1,0)^T$ .

The key is how to find the projection $X\hat{w}$ numerically rather than intuitively. Consider the vector $Y-X\hat{w}$, which starts from the end of $X\hat{w}$ pointing to the end of $Y$. $X\hat{w}$ is the projection if and only if $Y-X\hat w$ is perpendicular to all the column vectors of $X$. Therefore it follows that

$$\begin{aligned}X^T(Y-X\hat w)&=\mathbf{0}\\X^TX\hat w&=X^TY\\\hat w&=(X^TX)^{-1}X^TY,\end{aligned}$$

which is consistent with our conclusion in section 1. Also, plugging the values of the example above, we have $\hat w=(0.64, 1, 0.32)^T$, thus $X\hat w=(1,1,0)^T$, which is consistent with our (imaginary) observation of the projection.

> The term $(X^TX)^{-1}X^T$ is called [Moore–Penrose inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) of $X$.

# 3. Least Squares Method - A Probabilistic Perspective

As we mentioned before, error is inevitable in practice, otherwise there is no need to do such approximation.

## 3.1 Maximum Likelihood Estimation

We now use Gaussian distribution to reflect such noise as $\varepsilon\sim\mathcal{N}(0,\sigma^2)$. According the definition in section 1, we need to find a $w$ subject to

$$y=w^Tx+\varepsilon.$$

Obviously, we have $y\|x;w\sim\mathcal{N}(w^Tx,\sigma^2)$. For $N$ samples, the log-likelihood follows that

$$\begin{aligned}\mathcal{L}(w)&=\log P(Y|X;w)\\&=\sum_{i=1}^N\log P(y_i|x_i;w)\\&=\sum_{i=1}^N\left(\log \frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{2\sigma^2}(y_i-w^Tx_i)^2\right)\end{aligned}.$$

Therefore, by *maximum likelihood estimation* (MLE) method, we have

$$\begin{aligned}\hat w&=\arg\max_w\mathcal{L}(w)\\&=\arg\max_w -\sum_{i=1}^N\frac{1}{2\sigma^2}(y_i-w^Tx_i)^2\\&=\arg\min_w\sum_{i=1}^N(y_i-w^Tx_i)^2\end{aligned},$$

which is consistent with our analysis in section 1. In fact, the least squares method has an assumption that the noise follows a Gaussian distribution.

## 3.2 Maximum A Posteriori

As we mentioned in the [previous notes](https://2ez4ai.github.io/2020/09/28/intro-ml01/), in the view of Bayesians, $w$ can also be a random variable. Suppose $w\sim\mathcal{N}(0,\sigma_0^2)$. Still, we have $y\vert x;w\sim\mathcal{N}(w^Tx,\sigma^2)$ (there is a little abuse of notation: $w$ after ‘$\|$’ is a sample rather than a random variable). By *maximum a posteriori* (MAP) method, we have

$$\begin{aligned}\hat w&=\arg\max_w P(w|Y)\\&=\arg\max_w\log\left(\frac{\prod_{i=1}^NP(y_i|w)\cdot P(w)}{\prod_{i=1}^NP(y_i)}\right)\\&=\arg\max_w\log\left(\prod_{i=1}^NP(y_i|w)\right)+\log P(w)\\&=\arg\max_w\sum_{i=1}^N\log\left(\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}}\right)+\log\left(\frac{1}{\sqrt{2\pi}\sigma_0}e^{-\frac{||w||^2}{2\sigma_0^2}}\right)\\&=\arg\min_w\sum_{i=1}^N\frac{(y_i-w^Tx_i)^2}{2\sigma^2}+\frac{||w||^2}{2\sigma_0^2}\\&=\arg\min_w\underbrace{\sum_{i=1}^N(y_i-w^Tx_i)^2}_\text{square error}+\underbrace{\frac{\sigma^2}{\sigma_0^2}||w||^2}_\text{regularizer}\end{aligned},$$

which is slightly different from the result of MLE method. However, we will show in next section this is actually equivalent to the *regularized least squares method*.

# 4. Regularization

In practice, a common issue is $N\ll d$, which may cause $X^TX$ not invertible and further lead to *overfitting*. There are three techniques to avoid overfitting: collecting more data, *feature engineering/extracting*, and *regularization*. The regularization method is adding a penalty term, and we have the new loss function $\mathcal{L}_r(w)=\mathcal{L}(w)+\lambda P(w)$ where $\lambda$ is a tunable parameter. Depending on what $P(w)$ is, we have different regression.



**Lasso regression**: use $L1$ norm as the penalty, which means $P(w)=\|\|w\|\|$.

**Ridge regression**: use $L2$ norm as the penalty, which means $P(w)=\|\|w\|\|_2^2=w^Tw.$



Such regularization is often called *weight decay*. We now focus on ridge regression. According the above definition, we have

$$\begin{aligned}\mathcal{L}_r(w)&=\mathcal{L}(w)+\lambda P(w)\\&=\sum_{i=1}^N||w^Tx_i-y_i||^2+\lambda w^Tw\\&=w^TX^TXw-2w^TX^TY+Y^TY+\lambda w^Tw\\&=w^T(X^TX+\lambda\mathbf{I})w-2w^TX^TY+Y^TY\end{aligned}.$$

Therefore, we have the optimal weight where

$$\begin{alignat*}{3}\hat w&=\arg\min_w \mathcal{L}_r(w)\Rightarrow\frac{\partial \mathcal{L}_r(w)}{\partial w}&=0\end{alignat*}.$$

Solving the equation, we have

$$\hat{w}=(X^TX+\lambda\mathbf{I})^{-1}X^TY.$$

By introducing the positive definite matrix $\lambda\mathbf{I}$, the problem of the invertible matrix $X^TX$ is avoided.

# 5. Conclusion

Though linear regression is a naive model of machine learning, the thought of it is inspiring. In this post, we show that least squares is equivalent to MLE method with Gaussian noise in data, while the least squares with $L2$ regularizer is equivalent to MAP method with Gaussian noise in both weight and data.

Based on the attributes of linear regression, thoughts of many machine learning models can be derived:

- Linearity: as its name suggests, the linear regression method exploits linear functions to fit the data. Unsurprisingly, such linearity has a limited performance in general. Hence, there are many models developed from breaking the linearity. Examples:
  - **Polynomial Regression** is a form of linear regression in which we convert the original features into their higher order terms. For example, transforms $x=(x_1,x_2)$ into $\tilde x=(x_1,x_2,x_1x_2,x_2^3)$.
  - **Logistic Regression**: introduces a sigmoid function to the linear function, *e.g.* $f(x)=\text{sigmoid}(w^Tx)$.
  - **Neural Network** introduces multiple nonlinear functions and brings the multi-layer structure.

- Global Space: the approximation we found by linear regression is applied to the whole space. However, in practice, the data may not be continuous, and we need different approximations for different space.
  - **Decision Tree** divides the space into smaller sub-spaces depending on the question.
- Raw Data: the basic linear regression utilizes all the given data, which incurs the *curse of dimensionality*. In this case, dimensionality reduction methods are necessary.
  - **PCA** transforms the variables of the higher dimension into a smaller ones that still contain most of the information in the original data set.

