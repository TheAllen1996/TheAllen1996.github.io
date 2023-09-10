---
layout: post
data: 2020-11-05
tags: Machine-Learning
giscus_comments: True
title: Machine Learning - 06 Kernel Method
description: Introduce the basic concept of kernel methods and two properties of it with proof details. 
---

*The notes are based on the [session]( https://github.com/shuhuai007/Machine-Learning-Session). For the fundamental of linear algebra, one can always refer to [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/) and [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) for more details. Many thanks to these great works.*

* * what?
{:toc}
# 0. Introduction

Kernel methods are a class of algorithms for pattern analysis. The name of kernel methods comes from the use of *kernel function*, which enable operations in a high-dimensional and implicit space. Specifically, by [Cover's theorem](https://en.wikipedia.org/wiki/Cover%27s_theorem), given a set of training data that is not *linearly separable*, one can with high probability transform it into a training set that is linearly separable by projecting it into a higher-dimensional space via some *non-linear transformation*. With the help of kernel function, the operation, *i.e.,* inner product, it involves after transforming can be often computationally cheaper than the explicit computation. Such an approach is called the *kernel trick*. In this post, we will focus on the application of kernel method to SVM.

# 1. Kernel method

Define the data set as $$\mathcal{D}=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}, X=\{x_1,x_2,\dots,x_N\}$$ and $$Y=\{y_1,y_2,\dots,y_N\}$$ where $$x_i\in\mathbb{R}^{d\times 1}$$ and $$y_i\in\{-1,1\}$$. We further assume that the data set is non-linearly separable. Kernel method supposes that there is a non-linear transformation $$\phi(x):\mathbb{R}^{d\times 1}\to\mathbb{R}^{p\times 1},d<p,$$  such that $$\mathcal{D}_p=\{(\phi(x_1),y_1),(\phi(x_2),y_2),\dots,(\phi(x_N),y_N)\}$$ are linearly separable. For such a linearly separable data set, recalling the problem we formulated in section 1.3 of [SVM](https://2ez4ai.github.io/2020/10/28/support_vector_machine-ml05/), we have the duality problem

$$\begin{aligned}\min_\lambda&\quad \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^N\left(\lambda_i\lambda_jy_iy_j\phi^T(x_i)\phi(x_j)\right)-\sum_{i=1}^N\lambda_i\\\text{s.t.}&\quad \lambda_i\ge0,i=1,2,\dots,N\\&\quad \sum_{i=1}^N\lambda_iy_i=0.\end{aligned}$$

However, after transforming, the inner product $$\phi(x_i)^T\phi(x_j)=\langle\phi(x_i),\phi(x_j)\rangle$$ could be hard to obtain (consider the case that $$\phi(\cdot)$$ has infinite dimensions), which requires the aid of *kernel function*.

# 2. Kernel function

A kernel function is defined as $$K:\mathbb{R}^{d\times 1}\times\mathbb{R}^{d\times 1}\to\mathbb{R}$$. Specifically, for non-linear transformation $$\phi(\cdot)\in\mathcal{H}\text{ (Hilbert space) }:\mathbb{R}^{d\times 1}\to\mathbb{R}^{p\times 1}$$ and any $$x_i,x_j\in\mathbb{R}^{d\times 1}$$, we call $$K(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle$$ a kernel function. Such a kernel function is regarded *positive definite*, which satisfies

$$K(x_i,x_j)=K(x_j,x_i),$$

and for $$x_{1},x_{2},\dots,x_{N}\in\mathbb{R}^{d\times 1},$$

$$\mathcal{K}=[K(x_{i},x_{j})]_{N\times N}\text{ is a positive semi-definite (PSD) matrix},$$

where $$\mathcal{K}$$ is called *Gram matrix* of $$K$$ over set $$\{x_{1},x_{2},\dots,x_{N}\}$$. When the explicit expression of $$\phi(\cdot)$$ is hard to be determined, it quite often to show the positive definiteness of a kernel function via its corresponding Gram matrix. 

**(Properties)** We now show two *properties* of kernel functions. 

- Let $$K$$ be kernel function such that $$K:\mathbb{R}^{d\times 1}\times\mathbb{R}^{d\times 1}\to\mathbb{R}$$, then we define its Gram matrix $$\mathcal{K}\in\mathbb{R}^{N\times N}$$ over $$\{x_1,x_2,\dots,x_N\}$$ where $$x_i\in\mathbb{R}^{d\times 1}$$. Considering the mapping function $$\phi(\cdot):\mathbb{R}^{d\times 1}\to\mathbb{R}^{p\times 1}$$, we have

$$K(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle, \phi(\cdot)\in\mathcal{H}\implies \begin{cases}K(x_i,x_j)=K(x_j,x_i)\\ \mathcal{K}\text{ is a PSD matrix}\end{cases}.$$

*Proof*: 

By the definition of $$K(x_i,x_j)$$, we have

$$K(x_i,x_j)=\langle \phi(x_i),\phi(x_j)\rangle,\quad K(x_j,x_i)=\langle \phi(x_j),\phi(x_i)\rangle.$$

By the symmetry of inner product, we have $$\langle \phi(x_i),\phi(x_j)\rangle=\langle \phi(x_j),\phi(x_i)\rangle$$. It then follows that

$$K(x_i,x_j)=K(x_j,x_i).$$

Therefore the Gramian matrix $$\mathcal{K}=[K(x_{i},x_{j})]_{N\times N}$$ is symmetric real matrix. Now we show that $$\forall\alpha\in\mathbb{R}^{R\times 1}, \alpha^T\mathcal{K}\alpha\ge 0.$$ The notation is given by

$$\begin{aligned}\alpha^T\mathcal{K}\alpha=(\alpha_1,\alpha_2,\dots,\alpha_N)\begin{bmatrix}K_{11}&K_{12}&\dots&K_{1N}\\K_{21}&K_{22}&\dots&K_{2N}\\\vdots&\vdots&\ddots&\vdots\\K_{N1}&K_{N2}&\dots&K_{NN}\end{bmatrix}\begin{pmatrix}\alpha_1\\\alpha_2\\\vdots\\\alpha_N\end{pmatrix}\end{aligned},$$

where $$K_{ij}=K(x_{ri},x_{rj})$$. We then have

$$\begin{aligned}\alpha^T\mathcal{K}\alpha&=\sum_{i=1}^R\sum_{j=1}^R \alpha_i\alpha_jK_{ij}\\&=\sum_{i=1}^R\sum_{j=1}^R \alpha_i\alpha_j\phi^T(x_{ri})\phi(x_{rj})\\&=\sum_{i=1}^R \alpha_i\phi^T(x_{ri})\sum_{j=1}^R\alpha_j\phi(x_{rj})\\&=\left(\sum_{i=1}^R \alpha_i\phi(x_{ri})\right)^T\left(\sum_{j=1}^R\alpha_j\phi(x_{rj})\right)\\&=\left\langle\left(\sum_{i=1}^R \alpha_i\phi(x_{ri})\right), \left(\sum_{j=1}^R \alpha_i\phi(x_{rj})\right)\right\rangle\\&=\left\vert\left\vert\sum_{i=1}^R \alpha_i\phi(x_{ri})\right\vert\right\vert^2,\end{aligned}$$

therefore, $$\alpha^T\mathcal{K}\alpha\ge 0$$ and $$\mathcal{K}$$ is a PSD matrix.$$\tag*{$$\blacksquare$$}$$

- Let $$\mathcal{K}\in\mathbb{R}^{d\times d}$$ be a symmetric PSD matrix, then for $$\{x_1,x_2,\dots,x_N\}$$ where $$x_i\in\mathbb{R}^{d\times 1}$$, we have kernel function $$K(x_i,x_j)=x_i^T\mathcal{K}x_j$$.

*Proof*: 

Consider the *diagonalisation* of $$\mathcal{K}=Q^T\Lambda Q$$ by an orthogonal matrix $$Q$$, where $$\Lambda$$ is a diagnoal matrix containing the non-negative eigenvalues of $$\mathcal{K}$$. Let $$\sqrt{\Lambda}$$ be the diagonal matrix with the square roots of the eigenvalues and set $$A=\sqrt{\Lambda}Q$$.  Then for $$\{x_1,x_2,\dots,x_N\}$$ where $$x_i\in\mathbb{R}^{d\times 1}$$, we have

$$x_i^T\mathcal{K}x_j=x_i^TQ^T\Lambda Qx_j=x_i^TA^TA x_j=\langle A x_i,Ax_j\rangle.$$

Therefore we have kernel function $$K(x_i,x_j)=x_i^T\mathcal{K}x_j=\langle Ax_i,Ax_j\rangle$$ with linear transformation $$\phi(\cdot)=A\cdot. \tag*{$$\blacksquare$$}$$



# 3. Conclusion

In this post, we introduced *kernel method* for classification problem. Given *Cover’s theorem*, we can project non-linear data into high-dimensional space and obtain linearly separable data. To simplify the computation incurred by the duality problem, we can leverage *kernel function* to avoid the computing labor.

This is definitely not a good introduction to kernel methods. For more details of kernel method, I would recommend [Kernel methods: an overview](https://people.eecs.berkeley.edu/~jordan/kernels/0521813972c03_p47-84.pdf).