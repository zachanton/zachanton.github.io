---
layout: post
title: "Tropical Geometry of Deep Neural Networks"
date: 2021-01-06
description: "Looking at the relationship between tropical polynomials and neural networks."
img_url: /assets/img/tropical-deep-learning/new-boundaries.png
tags: [deep learning, tropical geometry,visualization]
language: [python]
comments: true
---

[Zhang et al.](https://arxiv.org/pdf/1805.07091.pdf) showed a close relationship between tropical polynomials and family of functions represented by feedforward neural networks with rectified linear units (ReLU) and integer weights. 
	
This connection helps to study both theoretical capabilities of neural networks, such as their expressiveness, and the methods of their  [pruning and initialization](https://arxiv.org/pdf/2002.08838.pdf). 

In this post we will take a closer look at the relationship between tropical polynomials and neural networks. An important point is the close geometric connection between tropical hypersurfaces and the decision boundaries of neural networks. We make use of the [code](https://github.com/zachanton/tropical) accompanying my [master's thesis](https://github.com/zachanton/tropical/blob/master/master_thesis.pdf), in order to visualize the decision boundary of a two-layer fully connected ReLU network,
trained for a binary classification problem.


**Outline**

- [Background](#background)
	- [Tropical Arithmetic and Tropical Linear Algebra](#TropicalArithmeticandTropicalLinearAlgebra)
	- [Tropical Algebraic Geometry](#TropicalAlgebraicGeometry)
	- [Neural Networks](#NeuralNetworks)
- [Tropical Polynomials and Neural Networks](#TropicalPolynomialsandNeuralNetworks)
- [Decision Boundaries and Tropical Geometry](#DecisionBoundariesandTropicalGeometry)
- [Conclusion](#conclusion)


## Tropical Arithmetic and Tropical Linear Algebra  <a name="TropicalArithmeticandTropicalLinearAlgebra"></a>

### Tropical Semiring

On the set \\(\mathbb{R} = R\cup\lbrace-\infty\rbrace\\) we define two commutative binary
operations \\(\oplus\\) and \\(\odot\\) as follows: for \\(a, b\in R\\) and
\\(c\in \mathbb{R}\\)

$$
a\oplus b = max(a, b),\; a\odot b = a + b
$$

$$
c\oplus -\infty = c,\; c\odot -\infty = -\infty
$$

The triplet \\(\mathbb{T} = \lbrace\mathbb{R}, \oplus, \odot\rbrace\\) is called the tropical
semiring.

Also we define *tropical quotient* of \\(x\\) and \\(y\\) as
\\(x \oslash y = x-y\\) and *tropical power* as
\\(x^{\odot a} = a \cdot x\\) for \\(a\in \mathbb{R}\\)

### Tropical Polynomial

Let \\(x_1,\dots,x_d\\) be variables representing elements of \\(\mathbb{T}\\). A *tropical monomial* is a finite product of any of these variables, with repetition allowed

$$
c \odot x_1^{\odot \alpha_1}\odot \dots \odot x_d^{\odot \alpha_d} = c + \alpha_1 x_1 + \dots + \alpha_d x_d
$$

A *tropical polynomial* is a finite sum of tropical monomials:

$$
c_1 \boldsymbol{x}^{\alpha_1} \oplus\dots\oplus c_n \boldsymbol{x}^{\alpha_n} = max(c_1+\langle \alpha_1,\boldsymbol{x} \rangle,\dots,c_n+\langle \alpha_n,\boldsymbol{x} \rangle)
$$

with \\(c_i\in\mathbb{T}\\) and \\(\boldsymbol{\alpha}_i = (\alpha_1,\dots,\alpha_d) \in Z^d \\) and a monomial of a given multiindex appears at most once in the sum, i.e., \\({\alpha}_i\neq{\alpha}_j\\) for any \\(i \neq j\\).
        
       
### Tropical Rational Map

A tropical quotient of two tropical polynomials \\(f(\\boldsymbol{x}) \\oslash g(\\boldsymbol{x})\\)
is a tropical rational function.

A map \\(F:\mathbb{R}^d\to\mathbb{R}^p:\boldsymbol{x}\mapsto(f_1(\boldsymbol{x}),\dots,f_p(\boldsymbol{x}))\\) is a *tropical polynomial map* if each \\(f_i(\boldsymbol{x})\\) can be represented by a tropical polynomial and a *tropical rational map* if each \\(f_i(\boldsymbol{x})\\) can be represented by a tropical rational function.

### Equivalence of tropical polynomials

There are two levels of equivalence
between tropical polynomials

- \\(f =_1 g:\;\\) \\(f\\) and \\(g\\) have the same terms and coefficients;
- \\(f =_2 g:\; f(x) = g(x)\\) for all \\(x \in R\\)

For example, the polynomials \\(x_1 \odot x_2 \oplus x_1^{2} \oplus x_2^{2} = max \left( x_1+x_2, 2 x_1, 2 x_2\right)\\) and \\(x_1^{2} \oplus x_2^{2}= max \left(2 x_1, 2 x_2\right)\\) are functionally equivalent, but not equal as polynomials. 

This follows from the fact that in the first polynomial monomial  \\(x_1 \odot x_2\\) is less than or equal to \\(x_1^{2}\\) for \\(x_2\leq x_1\\) and less than or equal to \\(x_2^{2}\\) for \\(x_1\leq x_2\\), which implies that its value at any point coincides with \\(x_1^{2} \oplus x_2^{2}\\).


## Tropical Algebraic Geometry <a name="TropicalAlgebraicGeometry"></a>

### Tropical Hypersurface

Let \\(f(\boldsymbol{x})\\) be a tropical polynomial. A point \\(\boldsymbol{x}\in R^n\\) is a root of \\(f\\) if the maximum is attained at least twice in the evaluation of \\(f(\boldsymbol{x})\\).

    
Let \\(f\\) be a tropical polynomial. The tropical hypersurface \\(\mathcal{T}(f)\\) is the set of all roots of \\(f\\).

Suppose \\(f(x, y) = x\oplus y\oplus 0\\), then \\(\mathcal{T}(f)\\) is on the figure.

![png](/assets/img/tropical-deep-learning/simple_curve.png)
    
### Newton Polytope

For a tropical polynomial \\(f(x) = c_1 x^{\alpha_1} \oplus\dots\oplus c_n x^{\alpha_n}\\), we define the *Newton polytope* of \\(f\\) as the convex hull of \\(\alpha_1,\dots,\alpha_n\\) in \\(\mathbb{R}^d\\)

$$
 \Delta(f) = \text{Conv}\{ \alpha_i : c_i \neq -\infty , i=1...n\}. 
$$

and the *extended Newton polytope* of \\(f\\) as
        
$$
\mathcal{P}(f) = \text{Conv}\{ (c_i,\alpha_i)\in\mathbb{R}^{d+1}:\alpha_i\neq-\infty \}
$$
    
The faces of extended Newton polytope \\(\mathcal{P}(f)\\) that are visible from above with respect to the first coordinate form the upper convex hull and called \\(\text{UF}(\mathcal{P}(f))\\).

### Dual Subdivision

Letting \\(\pi:\mathbb{R}^{d+1}\to\mathbb{R}^d\\) be the projection which drops the first coordinate, the *dual subdivision* of \\(f\\) is then the projection of the upper faces of \\(\mathcal{P}(f)\\), i.e.

$$
 \delta(f) = \pi(\text{UF}(\mathcal{P}(f))). 
$$

Below are shown the points of \\((1\odot x^2)\oplus(1\odot y^2)\oplus(2\odot 
xy)\oplus (2\odot x)\oplus (2\odot y)\oplus 1\\), their convex hull, the induced subdivision of the triangle and the dual tropical curve:

![png](/assets/img/tropical-deep-learning/conv_hull.png)
*source: arxiv.org/pdf/1908.07012.pdf*

## Neural Networks <a name="NeuralNetworks"></a>


An *\\(L\\)-layer feedforward neural network* is a map \\(v:\mathbb{R}^d\to\mathbb{R}^p\\)

$$
    v^{(L)} = \sigma^{(L)} \circ \rho^{(L)} \circ\dots\circ \sigma^{(1)} \circ \rho^{(1)} 
$$

with fixed *preactivation functions* \\(\rho^{(k)}(x) = W^{(k)}x + {b}^{(k)}\\) and usually non-linear *activation functions* \\(\sigma^{(k)}:\mathbb{R}^{n_k}\to\mathbb{R}^{n_k}\\).

*\\(ReLU\\) neural networks* is a class of neural networks \\(v:\mathbb{R}^d\to\mathbb{R}^p\\) with

$$
\sigma^{(k)}(x) = \text{max}\{x,t\}
$$

where \\(t \in \mathbb{R}\\) is called a threshold vector and max is taken coordinate-wise. Networks from this class are piecewise-linear functions.

Further we will restrict ourselves to integer weight matrices, i.e. \\(W^{(k)}\in\mathbb{Z}^{n_k\times n_{k-1}}\\)

This restriction is not very strict because one can always use approximation to rational numbers and clear denominators to obtain integer weights. 


## Tropical Polynomials and Neural Networks <a name="TropicalPolynomialsandNeuralNetworks"></a>

[Zhang et al.](https://arxiv.org/pdf/1805.07091.pdf) proved that ReLU neural networks and tropical rational functions are equivalent in the following sense:

- Let \\(v:\mathbb{R}^d\to\mathbb{R}^p\\).Then \\(v\\) can be defined by a tropical rational function if and only if \\(v\\) is a feedforward neural network under previous restrictions.
- A tropical rational function \\(F\oslash G\\) can be represented as an L-layer network with

$$
        L \leq\ \max{ 1 + [\log_2(r_F)], 1 + [\log_2(r_G)], [\log_2(d+1)] } + 1
$$

where \\(r_F\\) and \\(r_G\\) are the number of monomials in the tropical polynomials \\(F\\) and \\(G\\) respectively.

From the proof of this statement, we can extract an algorithm for converting a neural network into a tropical rational function and vice versa. 


### Neural Network to Tropical Rational Function

In this section we will define inductive procedure of conversion ReLU-network into quotient of two tropical polynomials. After that we transform this procedure into an algorithm.

Let's start with the following proposition:

Suppose \\(v(x) = F(x) \oslash G(x)\\) is a tropical rational map, \\(\rho(x) = A x + b\\) is a preactivation function with integer-valued matrix \\(A\\), and \\(\sigma(x)\\) is a ReLU activation function. Then there exist tropical polynomials \\(F'(x)\\) and \\(G'(x)\\) that \\(\sigma\circ \rho \circ v(x) = F'(x)\oslash G'(x)\\).

The proof can be found [here](https://github.com/zachanton/tropical/blob/master/master_thesis.pdf). This procedure can be transformed into an algorithm:

![png](/assets/img/tropical-deep-learning/algo1.png)


### Tropical Polynomial to Neural Network

In order to evaluate tropical polynomial at a point, we can, for example, find the value of each of the monomials, and then determine the maximum of the obtained values. These steps can be represented as a neural network: we will first construct a single layer computing the monomials, and then \\(\log_2(n)\\)layers computing their maximum.
Below we describe this construction in more details.

Given a tropical polynomial 

$$
f(x) =  c_1 x^{\alpha_1} \oplus\dots\oplus c_n x^{\alpha_n}
$$

we can define 

$$
\begin{align}
    \begin{split}
    c &= \begin{bmatrix}
           c_{1} \\
           c_{2} \\
           \vdots \\
           c_{n}
         \end{bmatrix}
    \end{split}
    \begin{split}
    A &= \begin{bmatrix}
           \boldsymbol{\alpha}_{1} \\
           \boldsymbol{\alpha}_{2} \\
           \vdots \\
           \boldsymbol{\alpha}_{n}
         \end{bmatrix}
      = \begin{bmatrix}
           \alpha_{1}^{1} & ... & \alpha_{1}^{n} \\
           \alpha_{2}^{1} & ... & \alpha_{2}^{n} \\
           \vdots & \ddots & \vdots \\
           \alpha_{n}^{1} & ... & \alpha_{n}^{n}
         \end{bmatrix}
    \end{split}
\end{align}
$$

Given a point \\(x = (x_1,...,x_n)\\), we can calculate the vector of monomials values as \\(y = Ax+c\\). Maximum of the coordinates of this vector is equal to the value of the polynomial \\(f\\) at the point \\(x\\).

Maximum of two numbers \\(y_1\\) and \\(y_2\\) can be found by performing the following procedure

1. Finding product \\(z\\) between matrix \\(W_1\\)
    and \\(y\\)
2. Finding \\(z'= \max(z,0)\\) coordinatewise
3. Product m of \\(z'\\) and \\(W_2\\) will be equal to desired maximum

where 

$$
\begin{align}
    \begin{split}
    W_1 &= \begin{bmatrix}
           1 & -1 \\
           0 &  1 \\
           0 & -1
         \end{bmatrix}
    \end{split}
    \begin{split}
    W_2 &= \begin{bmatrix}
           1 \\
           1 \\
           -1
         \end{bmatrix}
    \end{split}
    \begin{split}
    y &=\begin{bmatrix}
           y_1 \\
           y_2
         \end{bmatrix}
    \end{split}
\end{align}
$$

Generalization of this method allows finding the maximum of n-element vector \\(y\\). We can match each matrix multiplication with a linear layer and non-linear maximum with ReLU and construct corresponding neural network.

![png](/assets/img/tropical-deep-learning/algo2.png)


## Decision Boundaries and Tropical Geometry <a name="DecisionBoundariesandTropicalGeometry"></a>

### Decision Boundary

The *decision boundary* of a neural network with output \\(v = (v_1,\dots,v_p)\\) is the set of inputs that give (at least) two nodes with equal output i.e. 

$$
\mathcal{B}(v) = \{ x\in\mathbb{R}^d : v_i(x)=v_j(x) \text{ for some }i\neq j \}  
$$

### Decision Boundary from tropical perspective

For neural network

$$
v(x)= \begin{bmatrix}
                    v_{1}(x) \\
                    v_{2}(x) 
                \end{bmatrix}=
                \begin{bmatrix}
                    F_1(x)\oslash G_1(x) \\ 
                    F_2(x)\oslash G_2(x)
                \end{bmatrix}
$$

                
the decision boundary of \\(v\\) is contained in some tropical hypersurface determined by \\(v\\). Specifically

$$
\mathcal{B}(v) = \{ x\in\mathbb{R}^d : v_1(x)=v_2(x) \} \subseteq \mathcal{T}(R), 
$$

where \\(R(x)=F_1(x)\odot G_2(x)\oplus F_2(x)\odot G_1(x)\\) and \\(\mathcal{T}(R)\\) this is the tropical curve defined above.


In the [paper](https://arxiv.org/pdf/2002.08838.pdf), the authors state this proposition and discusses the decision  boundary  of  a  simple  neural  network  with Linear-ReLU-Linear architecture, which also satisfy our restrictions. The main results obtained in the paper are based on an analysis not of the decision  boundary itself, but of its superset \\(\mathcal{T}(R(x))\\). 

Below, using our *tropical* framework, we obtain an analytical expression directly for the decision boundary and demonstrate that it coincides with the estimated numerically. In addition, unlike [Alfarra](https://arxiv.org/pdf/2002.08838.pdf), the network of our interest can consist of an arbitrary number of hidden layers.

### Experiment

Let's create a standard classification task from *sklearn*


```python
X, Y = make_classification(n_samples=1000, n_features=2,
n_classes=2, n_redundant=0, random_state=5)
```

![png](/assets/img/tropical-deep-learning/5_class.png)

and train a simple Linear-ReLU-Linear neural network with 4 neurons in hidden layer.

After training, we can convert our neural network into a tropical rational function defined by 4 tropical polynomials \\(H_1(x), G_1(x), H_2(x), G_2(x)\\). We use the notation \\(H\\)instead \\(F\\) because the last layer of our network is not ReLU.

In this experiment we get

$$
\begin{align*}
    H_1(x) &= 20.0 \oplus 72.0 \odot x_1^{398} \odot x_2^{354} \oplus 88.0 \odot x_1^{374} \odot x_2^{306} \oplus 4.0 \odot x_1^{24} \odot x_2^{48}\\
    G_1(x) &= 247.0 \odot x_1^{62} \odot x_2^{314} \oplus -128.0 \odot x_1^{562} \odot x_2^{564} \oplus x_1^{24} \odot x_2^{48} \oplus -375.0 \odot x_1^{524} \odot x_2^{298}\\
    H_2(x) &= 246.0 \odot x_1^{80} \odot x_2^{350} \oplus -84.0 \odot x_1^{520} \odot x_2^{570} \oplus -331.0 \odot x_1^{482} \odot x_2^{304} \oplus -1.0 \odot x_1^{42} \odot x_2^{84}\\
    G_1(x) &= 28.0 \oplus 100.0 \odot x_1^{396} \odot x_2^{324} \oplus x_1^{42} \odot x_2^{84} \oplus 72.0 \odot x_1^{438} \odot x_2^{408}
\end{align*}
$$


![alt-text-1](/assets/img/tropical-deep-learning/5_h0_and_g0.png "Dual subdivision of \\(H_1\\) and \\(G_1\\)") ![alt-text-2](/assets/img/tropical-deep-learning/5_h1_and_g1.png "Dual subdivision of \\(H_2\\) and \\(G_2\\)")

The left figure shows the dual subdivisions
for the tropical polynomial \\(H_1\\) and \\(G_1\\) and the right for the tropical polynomial \\(H_2\\) and \\(G_2\\). It's interesting that \\(H_1\\) looks very similar to \\(G_2\\) and \\(G_1\\) to \\(H_2\\) and this similarity will remain in case of more complex polynomials.



## Conclusion <a name="conclusion"></a>

Quantile regressions are a great way to estimate uncertainty, but make sure your target variable is normally distributed and normalized!
