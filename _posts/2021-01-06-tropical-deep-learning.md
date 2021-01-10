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


### Decision Boundary

The *decision boundary* of a neural network with output \\(v = (v_1,\dots,v_p)\\) is the set of inputs that give (at least) two nodes with equal output i.e. 

$$
\mathcal{B}(v) = \{ x\in\mathbb{R}^d : v_i(x)=v_j(x) \text{ for some }i\neq j \}  
$$

## Tropical Polynomials and Neural Networks <a name="TropicalPolynomialsandNeuralNetworks"></a>

[Zhang et al.](https://arxiv.org/pdf/1805.07091.pdf) proved that ReLU neural networks and tropical rational functions are equivalent in the following sense:

- Let \\(v:\mathbb{R}^d\to\mathbb{R}^p\\).Then \\(v\\) can be defined by a tropical rational function if and only if \\(v\\) is a feedforward neural network under previous assumptions.
- A tropical rational function \\(F\oslash G\\) can be represented as an L-layer network with

$$
        \[L \leq\ \max{ 1 + [\log_2(r_F)], 1 + [\log_2(r_G)], [\log_2(d+1)] } + 1
$$

where \\(r_F\\) and \\(r_G\\) are the number of monomials in the tropical polynomials \\(F\\) and \\(G\\) respectively.

From the proof of this statement, we can extract an algorithm for converting a neural network into a tropical rational function and vice versa. 








## Conclusion <a name="conclusion"></a>

Quantile regressions are a great way to estimate uncertainty, but make sure your target variable is normally distributed and normalized!
