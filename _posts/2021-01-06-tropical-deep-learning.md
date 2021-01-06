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

On the set \\(\mathbb{R} = R\cup\{-\infty \}\\) we define two commutative binary
operations \\(\oplus\\) and \\(\odot\\) as follows: for \\(a, b\in R\\) and
\\(c\in \mathbb{R}\\)

$$
a\oplus b = max(a, b),\, a\odot b = a + b
$$

$$
c\oplus -\infty = c,\, c\odot -\infty = -\infty
$$

The triplet \\(\mathbb{T} = {\mathbb{R}, \oplus, \odot}\\) is called the tropical
semiring.

Also we define *tropical quotient* of \\(x\\) and \\(y\\) as
\\(x \oslash y = x-y\\) and *tropical power* as
\\(x^{\odot a} = a \cdot x\\) for \\(a\in \mathbb{R}\\)

### Tropical Polynomial

Let *x*<sub>1</sub>, …, *x*<sub>*d*</sub> be variables representing
elements of $\\T$. A *tropical monomial* is a finite product of any of
these variables, with repetition allowed
*c* ⊙ *x*<sub>1</sub><sup> ⊙ *α*<sub>1</sub></sup> ⊙ … ⊙ *x*<sub>*d*</sub><sup> ⊙ *α*<sub>*d*</sub></sup> = *c* + *α*<sub>1</sub>*x*<sub>1</sub> + … + *α*<sub>*d*</sub>*x*<sub>*d*</sub>

A *tropical polynomial* is a finite sum of tropical monomials:
$$\\begin{split}
                {}& c\_1 \\x^{\\alpha\_1} \\oplus\\dots\\oplus c\_n \\x^{\\alpha\_n} = max(c\_1+\\langle \\alpha\_1,\\x \\rangle,\\dots,c\_n+\\langle \\alpha\_n,\\x \\rangle)
            \\end{split}$$
with $c\_i\\in\\T$ and
$\\boldsymbol{\\alpha}\_i \\coloneqq (\\alpha\_{i1},\\dots,\\alpha\_{id})\\in\\Z^d$
and a monomial of a given multiindex appears at most once in the sum,
i.e., *α*<sub>*i*</sub> ≠ *α*<sub>*j*</sub> for any *i* ≠ *j*.

### Tropical Rational Map

A tropical quotient of two tropical polynomials $f(\\x) \\oslash g(\\x)$
is a tropical rational function.

A map
$F:\\R^d\\to\\R^p:\\boldsymbol{x}\\mapsto(f\_1(\\boldsymbol{x}),\\dots,f\_p(\\boldsymbol{x}))$
is a *tropical polynomial map* if each $f\_i(\\x)$ can be represented by
a tropical polynomial and a *tropical rational map* if each $f\_i(\\x)$
can be represented by a tropical rational function.

###Equivalence of tropical polynomials
 There are two levels of equivalence
between tropical polynomials

-   *f*=<sub>1</sub>*g*: *f* and *g* have the same terms and
    coefficients;

-   *f*=<sub>2</sub>*g*: *f*(*x*) = *g*(*x*) for all *x* ∈ *R*

The polynomials
*x*<sub>1</sub> ⊙ *x*<sub>2</sub> ⊕ *x*<sub>1</sub><sup>2</sup> ⊕ *x*<sub>2</sub><sup>2</sup> = *m**a**x*(*x*<sub>1</sub>+*x*<sub>2</sub>,2*x*<sub>1</sub>,2*x*<sub>2</sub>)
and
*x*<sub>1</sub><sup>2</sup> ⊕ *x*<sub>2</sub><sup>2</sup> = *m**a**x*(2*x*<sub>1</sub>,2*x*<sub>2</sub>)
are functionally equivalent, but not equal as polynomials.





## Conclusion

Quantile regressions are a great way to estimate uncertainty, but make sure your target variable is normally distributed and normalized!
