---
layout: post
title: "Tropical Geometry of Deep Neural Networks"
date: 2021-01-06
description: "Training gradient boosted decision trees with a quantile loss to predict taxi fares, in python using catboost and vaex."
img_url: /assets/img/tropical-deep-learning/new-boundaries.png
tags: [deep learning, tropical geometry,visualization]
language: [python]
comments: true
---

[Zhang et al.](https://arxiv.org/pdf/1805.07091.pdf) showed a close relationship between tropical polynomials and family of functions represented by feedforward neural networks with rectified linear units (ReLU) and integer weights. 
	
This connection helps to study both theoretical capabilities of neural networks, such as their expressiveness, and the methods of their  [pruning and initialization](https://arxiv.org/pdf/2002.08838.pdf). 
    
An important point is the close geometric connection between tropical hypersurfaces and the decision boundaries of neural networks.

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


## Tropical Arithmetic and Tropical Linear Algebra

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

- \\(f =_1 g\\): \\(f\\) and \\(g\\) have the same terms and coefficients;
- \\(f =_2 g\\): \\(f(x) = g(x)\\) for all \\(x \in R\\)

For example, the polynomials \\(x_1 \odot x_2 \oplus x_1^{2} \oplus x_2^{2} = max \left( x_1+x_2, 2 x_1, 2 x_2\right)\\) and \\(x_1^{2} \oplus x_2^{2}= max \left(2 x_1, 2 x_2\right)\\) are functionally equivalent, but not equal as polynomials. This follows from the fact that in the first polynomial monomial  \\(x_1 \odot x_2\\) is less than or equal to \\(x_1^{2}\\) for \\(x_2\leq x_1\\) and less than or equal to \\(x_2^{2}\\) for \\(x_1\leq x_2\\), which implies that its value at any point coincides with \\(x_1^{2} \oplus x_2^{2}\\).



## Conclusion

Quantile regressions are a great way to estimate uncertainty, but make sure your target variable is normally distributed and normalized!
