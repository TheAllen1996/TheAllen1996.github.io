---
layout: post
data: 2020-10-26
tags: Machine-Learning
giscus_comments: True
title: Machine Learning - 04 Dimensionality Reduction
description: Include PCA and its variants. 
---

*The notes are based on the [session]( https://github.com/shuhuai007/Machine-Learning-Session). For the fundamental of linear algebra, one can always refer to [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/) and [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) for more details. Many thanks to these great works.*

* what?
{:toc}
# 0. Curse of Dimensionality

*The following introduction is derived from [Pattern Recognition and Machine Learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf).*

High dimensionality often incurs not only calculation issue but also *overfitting*. The increasing in dimension can point to the methods becoming rapidly unwieldy and of limited practical utility.

Our geometrical intuitions, formed through a life spent in a space of three-dimension, can fail badly when we consider spaces of higher dimensionality. As a simple example, consider a sphere of radius $r = 1$ in a space of $d$ dimensions, and ask what is the fraction of the volume of the sphere that lies between radius $r=1-\varepsilon$ and $r=1$. We can evaluate this fraction by noting that the volume of a sphere of radius $r$ in $d$ dimensions must scale as $r^d$, and so we write

$$V_d(r)=K_dr^d,$$

where the constant $K_d$ depends only on $d$. Thus the required fraction is given by

$$\frac{V_d(1)-V_d(1-\varepsilon)}{V_d(1)}=\frac{K_dr^d-K_d(1-\varepsilon)^d}{K_dr^d}=1-(1-\varepsilon)^d.$$

Obviously, for large $d$, this fraction tends to 1 even for small values of $\varepsilon$. Thus, in spaces of high dimensionality, most of the volume of a sphere is concentrated in a thin shell near the surface!

[too lazy to clarify]: For examples, in a two-dimension space, consider a circle of radius $r$ in a square with side length $2r$.Then the ratio $K_2$ of the area of the circle to the area of the square is $$K_2=\frac{\pi r^2}{4r^2}=c_2,$$ where $c_2$ is a constant of two-dimension case. In three-dimension space, consider a sphere and a cube instead of the circle and the square. Then we have the ratio $K_3$ of the volume follows that $$K_3=\frac{\frac{4}{3}\pi r^3}{8r^3}=c_3.$$ Actually, the volume (or quality, or something else that involves $n$-dimensional metrics) of a sphere and a cube in $n$-dimension space is $cr^n$ and $2^nr^n$ (it may be easy to understand from the multiple integral perspective), respectively. Thus, when $n$ becomes large, it follows that $$\lim_{n\to\infty}K_n=\lim_{n\to\infty}\frac{cr^n}{2^nr^n}=0.$$

Luckily, for those data of high dimensionality, we can refer to *dimensionality reduction* algorithms.

# 1. Sample Mean and Variance

Before we move on, we give the following definitions. Suppose we have data $$X=(x_1,x_2,\dots,x_N)^T\in\mathbb{R}^{N\times d}$$, where $x_i\in\mathbb{R}^{d\times 1}$ is a sample with $d$ features. Specifically,

$$X=(x_1,x_2,\dots,x_N)^T=\begin{pmatrix}x_{1}^T\\x_{2}^T\\\vdots\\x_{N}^T\end{pmatrix}=\begin{pmatrix}x_{11}&x_{12}&\cdots&x_{1d}\\x_{21}&x_{22}&\cdots&x_{2d}\\\vdots&\vdots&\ddots&\vdots\\x_{N1}&x_{N2}&\cdots&x_{Nd}\end{pmatrix}_{N\times d}.$$

Then we have the **sample mean**

$$\bar X=\frac{1}{N}\sum_{i=1}^Nx_i,$$

and the **sample covariance**

$$S=\frac{1}{N}\sum_{i=1}^N\left(x_i-\bar X\right)\left(x_i-\bar X\right)^T.$$

However, the *sum* operation is inconvenient in calculating. Thus we further transform them into

$$\bar X=\frac{1}{N}(x_1,x_2,\dots,x_N)\begin{pmatrix}1\\1\\\vdots\\1\end{pmatrix}=\frac{1}{N}X^T\mathbf{1}_{N\times 1},$$

and

$$\begin{aligned}S&=\frac{1}{N}\underbrace{(x_1-\bar X,x_2-\bar X,\dots,x_N-\bar X)}_{X^T-\bar X(1,1,\dots,1)}\begin{pmatrix}(x_1-\bar X)^T\\(x_2-\bar X)^T\\\vdots\\(x_N-\bar X)^T\end{pmatrix}\\&=\frac{1}{N}\left(X^T-\bar X\mathbf{1}_{1\times N}\right)\left(X^T-\bar X\mathbf{1}_{1\times N}\right)^T\\&=\frac{1}{N}\left(X^T-\frac{1}{N}X^T\mathbf{1}_{N\times 1}\mathbf{1}_{1\times N}\right)\left(X^T-\frac{1}{N}X^T\mathbf{1}_{N\times 1}\mathbf{1}_{1\times N}\right)^T\\&=\frac{1}{N}X^T\underbrace{\left(\mathbf{I}_{N}-\frac{1}{N}\mathbf{1}_{N\times N}\right)}_{H}\left(\mathbf{I}_{N}-\frac{1}{N}\mathbf{1}_{N\times N}\right)^TX\\&=\frac{1}{N}X^THH^TX\\&=\frac{1}{N}X^THX,\end{aligned}$$

where $H$ is *centering matrix*, and it can be shown that $H^T=H,H^n=H$.

# 2. Principal Component Analysis

Principal component analysis (PCA) is defined as an [orthogonal](https://en.wikipedia.org/wiki/Orthogonal_transformation) [linear transformation](https://en.wikipedia.org/wiki/Linear_transformation) that transforms the data to a new coordinate system such that the greatest variance by some scalar projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.

Recall that in [the previous post](https://2ez4ai.github.io/2020/10/14/linear_classification-ml03/), we introduce *linear discriminant analysis* (LDA), which requires a projection that maximizes the distance between different classes. In PCA, instead of classes, we need to find a projection that maximizes the distance (variance) of all the data so that we can reduce the dimensionality (as a projection involves one dimensionality reduction) while losing the least information (as the greatest variance implies most information). Now we give the mathematically formulation.

The notation in section 1 will be used in this section as well. Since PCA is sensitive to the variance of the data, it is critical to normalize the variables over all dimensions, which yields

$z_i=x_i-\bar X.$

In many case, the values of $z_i\in\mathbb{R}^{d\times 1}$ should be scaled in $[0,1]$ but here we just consider the simple case. We then denote the transformation vector as $W=(w_1,w_2,\dots,w_d)\in\mathbb{R}^{d\times d}$ which is the *orthogonal basis* of the new coordinate system. Then the new data $\tilde z_i$, *i.e.*, the projection of $z_i$ on the new coordinate system is

$$\tilde z_i=z_i^TW=(z_i^Tw_1,z_i^Tw_2,\dots,z_i^Tw_d),$$

where $z_i^Tw_j\in\mathbb{R}$ is the value of $j$-th dimension of the new data. Now suppose we want reduce the dimension from $d$ to $k$, *i.e.,* we want to preserve the first $k$ dimensions of $\tilde z_i$ while maximizing the variance, which gives us the objective function

$$\mathcal{J}(w)=\sum_{j=1}^k\sum_{i=1}^N\frac{1}{N}\left(z_i^Tw_j-\overline{z^Tw_j}\right)^2,$$

where $$\sum_{i=1}^N\frac{1}{N}\left(z_i^Tw_j-\overline{z^Tw_j}\right)^2$$ is the variance of the new data in $j$-th dimension. $$\overline{z^Tw_j}$$ is the mean of the new data in $j$-th dimension, which is

$$\begin{aligned}\overline{z^Tw_j}&=\frac{1}{N}\sum_{i=1}^Nz_i^Tw_j\\&=\frac{1}{N}\sum_{i=1}^N(x_i-\bar X)^Tw_j\\&=\frac{1}{N}\left(\sum_{i=1}^Nx_i^Tw_j-N\bar X^Tw_j\right)\\&=\frac{1}{N}\left(\sum_{i=1}^Nx_i^Tw_j-\sum_{i=1}^Nx_i^Tw_j\right)\\&=0\end{aligned}.$$

Therefore the objective function follows that

$$\begin{aligned}\mathcal{J}(w)&=\sum_{j=1}^k\sum_{i=1}^N\frac{1}{N}\left(z_i^Tw_j\right)^2\\&=\sum_{j=1}^k\sum_{i=1}^N\frac{1}{N}\left((x_i-\bar X)^Tw_j\right)^2\\&=\sum_{j=1}^k\sum_{i=1}^N\frac{1}{N}w_j^T(x_i-\bar X)(x_i-\bar X)^Tw_j\\&=\sum_{j=1}^kw_j^TSw_j\end{aligned},$$

where each $w_j^TSw_j$ can be maximized independently since $w_j$ are orthogonal. Combined with the constraint that $w_j$ is a unit vector, the problem is then equivalent to

$$\hat w_j=\arg\max_{w_j} w_j^TSw_j\quad \text{s.t. }w_j^Tw_j=1.$$

The problem can be solved by [the method of Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier) as follows,

$$\begin{aligned}\mathcal{L}(w_j,\lambda_j)&=w_j^TSw_j+\lambda_j(1-w_j^Tw_j),\\\frac{\partial\mathcal{L}(w_j,\lambda_j)}{\partial w_j}&=2Sw_j-2\lambda_j w_j.\end{aligned}$$

Setting the second equation to be zero, we have the standard eigenvalue problem

$$S\hat{w}_j=\lambda_j\hat{w}_j\implies SW=\text{diag}(\lambda)W ,$$

*i.e.,* $\hat w_j$ is actually the eigenvalue of $S$. Plugging the above equation to the objective function, we have

$$\begin{aligned}\max\mathcal{J(\hat{w})}&=\max\sum_{j=1}^k\hat{w}_j^TS\hat{w}_j\\&=\max\sum_{j=1}^k\hat{w}_j^T\lambda_j\hat{w}_j\\&=\max\sum_{j=1}^k\lambda_j\end{aligned}.$$

Therefore, to reduce the data from $d$ to $k$ dimension, one should select the $k$ eigenvectors of $S$ corresponding to the $k$ largest eigenvalues to construct the $W\in\mathbb{R}^{d\times k}$, which is

$$W=(w_{(1)},w_{(2)},\dots,w_{(k)}),$$

where $w_{(i)}$ is the eigenvector corresponding to the $i$-th largest eigenvalue.  Then the data in the new coordinate system is

$$X^\text{new}=(x_1-\bar X,x_2-\bar X,\dots, x_N-\bar X)^TW.$$

# 3. Principal Component Analysis - An SVD Perspective

Now we consider the *singular vector decomposition* (SVD) of the centralized data,

$$HX=U\Sigma V^T,$$

where $U=(u_1,u_2,\dots,u_N)\in\mathbb{R}^{N\times N},\Sigma\in\mathbb{R}^{N\times d},V=(v_1,v_2,\dots,v_N)\in\mathbb{R}^{d\times d}$, and they have the following properties,

$$\begin{aligned}UU^T&=U^TU=\mathbf{I}_{N},\\VV^T&=V^TV=\mathbf{I}_{d},\\\Sigma_{ij}=0,\quad i&=0,1,…,N,j=0,1,…,d,i\ne j.\end{aligned}$$

We further represent $\Sigma$ as $\lambda(\sigma_1,\sigma_2,\dots,\sigma_d)$. Then according to the analysis in section 1, the covariance of the data is

$$\begin{aligned}S&=\frac{1}{N}X^THX\\&=\frac{1}{N}X^TH^THX\\&=\frac{1}{N}(HX)^THX\\&=\frac{1}{N}V\Sigma^TU^TU\Sigma V^T\\&=V\text{diag}\left(\frac{\sigma_1^2}{N},\frac{\sigma_2^2}{N},\dots,\frac{\sigma_d^2}{N}\right)V^T\end{aligned}.$$

By multiplying $V$ on both sides, it follows that

$$SV=V\text{diag}\left(\frac{\sigma_1^2}{N},\frac{\sigma_2^2}{N},\dots,\frac{\sigma_d^2}{N}\right)\implies Sv_i=\frac{\sigma_i^2}{N}v_i,$$

which is consistent with the conclusion of PCA. Therefore, instead of decomposing the covariance matrix $S$, we can conduct an SVD on the centralized data, which gives the transformation matrix that allows us to obtain the new data

$$Z=HX\cdot V.$$

By selecting $k$ vectors of $V$ according to the single values, we can reduce the original data matrix from $d$ to $k$ dimensions. Such decomposition may take advantage when $N\ll d$. Intuitively, $HX\in\mathbb{R}^{N\times d}$ and $S\in\mathbb{R}^{d\times d}$. When $N\ll d$, decomposing $HX$ should be more efficient than decomposing $S$.

# 4. Principal Coordinates Analysis

*Principal coordinates analysis* (PCoA) is a well known technique in many fields. It actually can be derived from PCA. Specifically, we consider a matrix

$$T=HXX^TH^T.$$

By SVD, we have

$$T=U\Sigma V^TV\Sigma^TU^T=U\Sigma\Sigma^T U^T,$$

which is similar to the decomposition of $S$ in section 3. Further, by multiplying $U\Sigma$ on the both sides, we have

$$TU\Sigma=U\Sigma\Sigma^TU^TU\Sigma=U\Sigma\text{diag}\left(\sigma_1^2,\sigma_2^2,\dots,\sigma_d^2\right)\implies T(U\Sigma)_i=\sigma^2_i(U\Sigma)_i.$$

Therefore, $U\Sigma$ is actually composed of $d$  eigenvalues of $T$. Recall that in section 3, we have the new data as

$$Z=HX\cdot V=U\Sigma V^T\cdot V=U\Sigma,$$

which implies that by the eigenvalue decomposition of $T$, we can get the new data directly. Such a dimensionality reduction technique with a different perspective is PCoA. Notice $T\in\mathbb{R}^{N\times N}$, thus the complexity of PCoA is $O(N^2)$.

# 5. Probabilistic Principal Component Analysis

Just like the notations we used in previous sections, we define the new data we want to transform $X$ into is $Z=(z_1,z_2,\dots,z_N)$ where $z_i\in\mathbb{R}^{k\times 1}$. However, in *probabilistic principal component analysis* (PPCA), we further introduce randomness as

$$z_i\sim\mathcal{N}(\mathbf{0}_{k\times 1}, \mathbf{I}_{k\times k}),$$

$$x_i=Wz_i+\mu+\varepsilon,$$

$$\varepsilon\sim\mathcal{N}(\mathbf{0}_{d\times 1},\sigma^2\mathbf{I}_{d\times d}),$$

where $W\in\mathbb{R}^{d\times k}, \mu\in\mathbb{R}^{d\times 1}$ and $\sigma^2$ are the parameters to be learned ($\mu$ can be viewed as the bias term of many machine learning model). Such randomization can generalize the model to the unseen data. One can also refer such a model to *linear Gaussian model*. In particular,  there are two phases. The first is learning:

$$\hat W,\hat \mu, \hat \sigma^2=\arg\max_{W,\mu,\sigma^2}P(X\vert Z;W,\mu,\sigma^2).$$

The second is inference (dimensionality reduction):

$$Z=\arg\max_{\tilde Z} P(\tilde Z\vert X).$$

The learning part can be dealt with *expectation–maximization algorithm*. The following is the details of the inference procedure. According to the definition, we have

$$x_i\vert z_i\sim\mathcal{N}(Wz_i+\mu,\sigma^2\mathbf{I}_{d\times d}),$$

where $z_i$ is a sample rather than a random variable. Also, we have

$$\mathbb{E}[x_i]=\mathbb{E}[Wz_i+\mu+\varepsilon]=\mu,$$

$$var[x_i]=var[Wz_i+\mu+\varepsilon]=var[Wz_i]+var[\varepsilon]=WW^T+\sigma^2\mathbf{1}_{d\times d},$$

and

$$x_i\sim\mathcal{N}(\mu, WW^T+\sigma^2\mathbf{1}_{d\times d}).$$

Then we consider the covariance $Cov(x_i,z_i)$,

$$\begin{aligned}Cov(x_i,z_i)&=\mathbb{E}[(x_i-\mu)(z_i-\mathbf{0}_{k\times 1})^T]\\&=\mathbb{E}[(x_i-\mu)z_i^T]\\&=\mathbb{E}[(Wz_i+\varepsilon)z_i^T]\\&=\mathbb{E}[Wz_iz_i^T]+\mathbb{E}[\varepsilon z_i^T]\\&=Wvar[z_i]+\mathbb{E}[\varepsilon]\mathbb{E}[z_i^T]\\&=W\end{aligned}.$$

Hence the union distribution of $(x_i,z_i)^T$ is

$$\begin{pmatrix}x_i\\z_i\end{pmatrix}\sim\mathcal{N}\left(\begin{pmatrix}\mu\\\mathbf{0}_{k\times1} \end{pmatrix},\begin{pmatrix}WW^T+\sigma^2\mathbf{1}_{d\times d}&W\\W^T&\mathbf{1}_{k\times k}\end{pmatrix}\right).$$

By the formula derived in [session 01](https://2ez4ai.github.io/2020/09/28/intro-ml01/), it can be shown that

$$z_i\vert x_i\sim\mathcal{N}(W^T(WW^T+\sigma^2\mathbf{1}_{d\times d})^{-1}(x_i-\mu),\mathbf{1}_{k\times k}-W^T(WW^T+\sigma^2\mathbf{1}_{d\times d})^{-1}W).$$

# 6. Conclusion

In this post, we introduced the naive *principal component analysis* (PCA) model. Then we conducted *singular vector decomposition* on the centralized data, which gives us the same conclusion as that of PCA. Such conclusion can also be derived from *Principal coordinates analysis* (PCoA) model. Further, by introducing parameters, *probabilistic principal component analysis* (PPCA) can generalize PCA to handle unseen data. 

