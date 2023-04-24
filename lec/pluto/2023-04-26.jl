### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ bcccf780-e24f-11ed-0075-7d1896fdc11f
using LinearAlgebra

# ╔═╡ bcccf7b2-e24f-11ed-244e-bf7a0eb3c9d2
using Plots

# ╔═╡ bcccf7bc-e24f-11ed-1b76-29e34e1bd44d
using LinearMaps

# ╔═╡ bcccf7c4-e24f-11ed-3c04-bd6d278d695c
using IterativeSolvers

# ╔═╡ bcccf69a-e24f-11ed-14a8-85b0acb3b500
md"""
# Notebook for 2023-04-26
"""

# ╔═╡ bcccf956-e24f-11ed-1f7c-0925bcb8773c
md"""
## Computing with constraints

Recall that our basic problem is

$$\mbox{minimize } \phi(x) \mbox{ s.t. } x \in \Omega$$

where the feasible set $\Omega$ is defined by equality and inequality conditions

$$\Omega = \{ x \in {\mathbb{R}}^n : c_i(x) = 0, i \in \mathcal{E} \mbox{ and }
    c_i(x) \leq 0, i \in \mathcal{I} \}.$$


In the last lecture, we described three different ways to formulate
constrained optimization problem that allow us to build on techniques
we previously explored from unconstrained optimization and equation
solving:

1.  *Constraint elimination* (for equality constraints): Find a
    parameterization $g : {\mathbb{R}}^{n-m} \rightarrow \Omega$
    formulations and minimize $\phi(g(y))$ without constraints. This
    requires that the constraints be simple (e.g. affine equality
    constraints).
2.  *Barriers and penalties:* Add a term to the objective function
    depending on some parameter $\mu$. This term penalizes $x$ values
    that violate the constraint (penalty methods) or that come close to
    $\partial \Omega$ from the inside (barrier methods). As
    $\mu \rightarrow 0$, the unconstrained minimum of the modified
    problems converges to the constrained minimum of the original.
3.  *Lagrange multipliers*: Add new variables (multipliers)
    corresponding to “forces” needed to enforce the constraints. The
    *KKT conditions* are a set of nonlinear equations in the
    original unknowns and the multipliers that characterize constrained
    stationary points.

Our goal now is to sketch how modern constrained optimization algorithms
incorporate these different ways of looking at the problem. A full
treatment is well beyond the scope of the class, but we hope to give you
at least the keywords you will need should you encounter them in a
textbook, paper, or a cocktail party. Ideally, knowing something
about what happens in the algorithms will also help you think about
which of various equivalent formulations of an optimization problem will
be more (or less) friendly to solvers. The plan is to first give a "lay
of the land" of different families of algorithms, then to give a more
detailed treatment with the running example of linearly constrained
quadratic programs.

For more details, there are some excellent textbooks in the field;
some texts that I really like from my own shelf include:

- [*Numerical Optimization*, Nocedal and Wright](https://link.springer.com/book/10.1007/978-0-387-40065-5)
- [*Practical Optimization*, Gill, Murray, and Wright](https://doi.org/10.1137/1.9781611975604)
- [*Nonlinear Programming*, Bertsekas](http://www.athenasc.com/nonlinbook.html)
"""

# ╔═╡ bcccfb06-e24f-11ed-1ed3-e9762b9d18f8
md"""
## Lay of the Land

As we mentioned before, problems with *inequality* constraints tend
to be more difficult than problems with *equality* constraints
alone, because it involves the combinatorial subproblem of figuring out
which constraints are *active* (a constraint $c_i(x) \leq 0$ is
active if $c_i(x) = 0$ at the optimum). Once we have figured out the set
of active constraints, we can reduce an inequality-constrained problem
to an equality-constrained problem. Hence, the purely
equality-constrained case is an important subproblem for
inequality-constrained optimizers, as well as a useful problem class in
its own right.

For problems with only equality constraints, there are several standard
options:

- *Null space methods* deal with linear equality constraints by
  reducing to an unconstrained problem in a lower-dimensional space.
- *Projected gradient methods* deal with simple equality
  constraints by combining a (scaled) gradient step and a projection
  onto a constraint set.
- *Penalty methods* approximately solve an equality-constrained
  problem through an unconstrained problem with an extra term that
  penalizes proposed soutions that violate the constraints. That is,
  we use some constrained minimizer to solve
  
$$\mbox{minimize } \phi(x) + \frac{1}{\mu} \sum_{i \in\mathcal{E}} c_i(x)^2.$$

  As $\mu \rightarrow 0$, the minimizers to these approximate problems
  approach the true minimizer, but the Hessians that we encounter
  along the way become increasingly ill-conditioned (with condition
  number proportional to $\mu^{-1}$).
- *KKT solvers* directly tackle the first-order optimality
  conditions (the KKT conditions), simultaneously computing the
  constrained minimizer and the associated Lagrange multipliers.
- *Augmented Lagrangian* methods combine the advantages of penalty
  methods and the advantages of the penalty formulation. In an
  augmented Lagrangian solver, one finds critical points for the
  augmented Lagrangian 
  
$$\mathcal{L}(x, \lambda; \mu) =
  \phi(x) + \frac{1}{\mu} \sum_{i \in \mathcal{E}} c_i(x)^2 + \lambda^T c(x)$$

  by alternately adjusting the penalty parameter $\mu$ and the
  Lagrange multipliers.

In the inequality-constrained case, we have

- *Active set methods* solve (or approximately solve) a sequence
  of equality-constrained subproblems, shuffling constraints into and
  out of the proposed working set along the way. These methods are
  particularly attractive when one has a good initial estimate of the
  active set.
- *Projected gradient methods* deal with simple inequality
  constraints by combining a (scaled) gradient step and a projection
  onto a constraint set.
- *Barrier methods* and *penalty methods* add a term to the
  objective function in order to penalize constraint violations or
  near-violations; as in the equality-constrained case, a parameter
  $\mu$ governs a tradeoff between solution quality and conditioning
  of the Hessian matrix.
- *Interior point methods* solve a sequence of barrier subproblems
  using a continuation strategy, where the barrier or penalty
  parameter $\mu$ is the continuation parameter. This is one of the
  most popular modern solver strategies, though active set methods may
  show better performance when one “warm starts” with a good initial
  guess for the solution and the active set of constraints.

As with augmented Lagrangian strategies in the equality-constrained
case, state-of-the art strategies for inequality-constrained problems
often combine approaches, using continuation with respect to a barrier
parameters as a method of determining the active set of constraints in
order to get to an equality-constrained subproblem with a good initial
guess for the solution and the Lagrange multipliers.

The *sequential quadratic programming* (SQP) approach for nonlinear
optimization solves a sequence of linearly-constrained quadratic
optimization problems based on Taylor expansion of the objective and
constraints about each iterate. This generalizes simple Newton iteration
for unconstrained optimization, which similarly solves a sequence of
quadratic optimization problems based on Taylor expansion of the
objective. Linearly-constrained quadratic programming problems are hence
an important subproblem in SQP solvers, as well as being an important
problem class in their own right.
"""

# ╔═╡ bcccfb90-e24f-11ed-1559-390ce7575c03
md"""
## Quadratic programs with equality constraints

We begin with a simple case of a quadratic objective and linear equality
constraints: 

$$\begin{align*}
  \phi(x) &= \frac{1}{2} x^T H x - x^T d \\
  c(x) &= A^T x-b = 0,
\end{align*}$$

where $H \in {\mathbb{R}}^{n \times n}$ is symmetric and positive definite
*on the null space of $A^T$* (it may be indefinite or singular
overall), $A \in {\mathbb{R}}^{n \times m}$ is full rank with $m < n$,
and $b \in {\mathbb{R}}^m$. Not only are such problems useful in their
own right, solvers for these problems are also helpful building blocks
for more sophisticated problems — just as minimizing an unconstrained
quadratic can be seen as the starting point for Newton’s method for
unconstrained optimization.
"""

# ╔═╡ bcccfb98-e24f-11ed-1281-3b2d11300cfb
begin
    # Set up a test problem for linearly-constrained QP (2D so that we can plot)
    H = [4.0  1.0 ;
    	 1.0  4.0 ]
    d = [0.5 ; -2.0]
    A = [1.0 ; 1.0]
    b = [1.0]
    
    ϕ1(xy) = xy'*H*xy/2 - xy'*d
    c1(xy) = A'*x - b
end

# ╔═╡ bcccfbae-e24f-11ed-3de0-817dbc26f386
let
    xx = range(-3, 3, length=100)
    plot(xx, xx, (x,y) -> ϕ1([x; y]), st=:contour, legend=false)
    plot!(xx, 1.0 .- xx, linewidth=2)
end

# ╔═╡ bcccfc6e-e24f-11ed-3ea1-5d397d767757
md"""
### Constraint elimination (linear constraints)

As discussed last time, we can write the space of solutions to the
constraint equations in terms of a (non-economy) QR decomposition of
$A$:

$$A =
  \begin{bmatrix} Q_1 & Q_2 \end{bmatrix}
  \begin{bmatrix} R_1 \\ 0 \end{bmatrix}$$

where $Q_2$ is a basis for the null space of $A^T$. The set of
solutions satisfying the constraints $A^T x = b$ is

$$\Omega = \{ u + Q_2 y : y \in {\mathbb{R}}^{(n-m)}, u = Q_1 R_1^{-T} b \};$$

here $u$ is a *particular solution* to the problem. If we substitute
this parameterization of $\Omega$ into the objective, we have the
unconstrained problem 

$$\mbox{minimize } \phi(u + Q_2 y).$$

While we can substitute directly to get a quadratic objective in terms of $y$, 
it is easier (and a good exercise in remembering the chain rule) to compute
the stationary equations

$$\begin{align*}
  0
  &= \nabla_y \phi(u + Q_2 y) 
  = \left(\frac{\partial x}{\partial y}\right)^T \nabla_x \phi(u+Q_2 y) \\
  &= Q_2^T (H (Q_2 y + u) - d) 
  = (Q_2^T H Q_2) y - Q_2^T (d-Hu).
\end{align*}$$

In general, even if $A$ is sparse, $Q_2$ may be dense, and so even if
$H$ is dense, we find that $Q_2^T H Q_2$ is dense.
"""

# ╔═╡ bcccfc76-e24f-11ed-0404-bb8cc740f1cd
begin
    # Solve the 2-by-2 problem via a null-space approach
    F = qr(A)
    Q = F.Q * I
    Q1 = Q[:,[1]]
    Q2 = Q[:,[2]]
    
    u_ns = Q1*(F.R'\b)
    H22 = Q2'*H*Q2
    r2  = Q2'*(d-H*u_ns)
    y_ns   = H22\r2
    x_ns   = u_ns + Q2*y_ns
end

# ╔═╡ bcccfc80-e24f-11ed-0e99-6f4f4a9d0920
let
    xx = range(-3, 3, length=100)
    plot(xx, xx, (x,y) -> ϕ1([x; y]), st=:contour, legend=false)
    plot!(xx, 1.0 .- xx, linewidth=2)
    plot!([u_ns[1]], [u_ns[2]], markercolor=:white, marker=true)
    plot!([x_ns[1]], [x_ns[2]], marker=true)
end

# ╔═╡ bcccfd0c-e24f-11ed-1b3c-7dd8ce4dd14b
md"""
Finding a particular solution and a null space basis via QR is great
for numerical stability, but it may not be ideal when the matrices
involved are sparse or structured.  An alternative is to use a sparse
LU factorization of $A^T$:

$$P A^T Q = L \begin{bmatrix} U_1 U_2 \end{bmatrix}.$$

where the $U_1$ submatrix is upper triangular.  A particular solution is then

$$x = Q \begin{bmatrix} U_1^{-1} L^{-1} P b \\ 0 \end{bmatrix}$$

and the null space is spanned by

$$Q^T 
  \begin{bmatrix}
    -U_1^{-1} U_2 \\
    I
  \end{bmatrix}.$$


This reformulation may be particularly attractive if $A$ is large,
sparse, and close to square.  Note that pivoting on rectangular
constraint matrices needs to be done carefully, e.g. using so-called
*rook pivoting* strategies that maintain numerical stability on
full-rank matrices with rank-deficient submatrices.
"""

# ╔═╡ bcccfe4c-e24f-11ed-218d-ad5aaa45060c
md"""
### Projected gradient and conjugate gradients

The *projected gradient* is a variant of gradient descent for
constrained problem.  One assumes that we have a projection operator
$P$ such that $P(x)$ is the closest point to $x$ satisfying the
constraint; the iteration is then

$$x_{k+1} = P\left( x_k - \alpha_k \nabla \phi(x_k) \right).$$

That is, we take an (unconstrained) gradient descent step, then
project back to satisfy the constraint.  It's an easy enough method to
code, provided you have the projection $P$.

For our linear equality constraints the projection can be computed by
a least squares type of solve:

$$\begin{align*}
  P(x) &= x + A(A^T A)^{-1} (b-A^T x) \\
       &= (A^T)^\dagger b + (I-AA^\dagger) x \\
       &= (A^T)^\dagger b + (I-\Pi) x
\end{align*}$$

Note that $(A^T)^\dagger b$ is the minimal norm solution to the
constraint equation, and the range space of $I-\Pi = I-AA^\dagger$ is
the null space of $A^T$, so this is similar to the picture we saw with
the constraint elimination approach.  And, of course, the gradient in
this case is just the residual $r_k = Hx_k - d$.

If we start with a point $x_0$ that is consistent with the
constraints, then each successive point remains on our linear
constraint surface; in this case, we can simplify the iteration to

$$x_{k+1} = x_k - \alpha_k (I-\Pi) r_k.$$

This is a stationary iteration for the underdetermined consistent equation

$$(I-\Pi) (Hx_k-d) = 0.$$

Unfortunately, the projected gradient iteration may converge rather
slowly.  A tempting thought is to use a scaled version of the
gradient, but the naive version of this iteration will in general
converge to the wrong point unless the projection operator is
re-defined in terms of the distance associated with the same scaling
matrix.

If the relevant projection is available, a potentially more attractive
route for this problem is to write $x = u + z$ for some particular
solution $u$ (as in the null space approach) and then use a method
like conjugate gradients on the system

$$(I-\Pi) H (I-\Pi) z = (I-\Pi) (d - Hu).$$

It turns out that the Krylov subspace generated by this iteration
remains consistent with the constraint, and so -- somewhat
surprisingly at first glance -- the method continues to work even
though $(I-\Pi) H (I-\Pi)$ is singular.
"""

# ╔═╡ bcccfe56-e24f-11ed-2580-45a4e0e55a94
let
    resid_proj(z) = z-Q1*(Q1'*z)  # Function to apply I-Π
    rhs = resid_proj(d-H*u_ns)    # Compute (I-Π)(d-Hu)

    # Define a 2-by-2 symmetric linear map (I-Π)H(I-Π) from a matvec function
    Afun = LinearMap((z)->resid_proj(H*resid_proj(z)), 2, issymmetric=true)
    
    # Solve the singular system via CG
    u_ns + cg(Afun, rhs)
end

# ╔═╡ bcccff8c-e24f-11ed-0118-ed93112b26d7
md"""
### Penalties and conditioning

Now consider a penalty formulation of the same equality-constrained
optimization function, where the penalty is quadratic:

$$\mbox{minimize } \phi(x) + \frac{1}{2\mu} \|A^T x-b\|^2.$$

In fact, the augmented objective function is again quadratic, and the critical
point equations are

$$(H + \mu^{-1} AA^T) x = d + \mu^{-1} A b.$$

If $\mu$ is small enough and the equality-constrained quadratic program (QP)
has a minimum, then $H+\mu^{-1} AA^T$ is guaranteed to be positive definite.
This means we can solve via Cholesky; or (if the linear system is larger)
we might use conjugate gradients.

We can analyze this more readily by changing to the $Q$ basis from the QR
decomposition of $A$ that we saw in the constraint elimination approach:

$$\begin{bmatrix}
  Q_1^T H Q_1 + \mu^{-1} R_1 R_1^T & Q_1^T H Q_2 \\
  Q_2^T H Q_1 & Q_2^T H Q_2
\end{bmatrix}
(Q^T x) =
\begin{bmatrix}
  Q_1^T d + \mu^{-1} R_1 b \\
  Q_2^T d
\end{bmatrix}$$

Taking a Schur complement, we have

$$(\mu^{-1} R_1 R_1^T + F)(Q_1^T x) = \mu^{-1} R_1 b - g$$

where

$$\begin{align*}
  F &= Q_1^T H Q_1 - Q_1^T H Q_2 (Q_2^T H Q_2)^{-1} Q_2^T H Q_1 \\
  g &= [I - Q_1^T H Q_2 (Q_2^T H Q_2)^{-1} Q_2^T] d
\end{align*}$$

As $\mu \rightarrow 0$, the first row of equations is dominated by the
$\mu^{-1}$ terms, and we are left with

$$R_1 R_1^T (Q_1^T x) - R_1 b \rightarrow 0$$

i.e. $Q_1 Q_1^T x$ is converging to $u = Q_1 R_1^{-T} b$, the particular
solution that we saw in the case of constraint elimination. Plugging this
behavior into the second equation gives

$$(Q_2^T H Q_2) (Q_2^T x) - Q_2^T (d-Hu) \rightarrow 0,$$

i.e. $Q_2^T x$ asymptotically behaves like $y$ in the previous example.
We need large $\mu$ to get good results if the constraints are ill-posed or if
$Q_2^T H Q_2$ is close to singular. But in general the condition number
scales like $O(\mu^{-1})$, and so large values of $\mu$ correspond to
problems that are numerically unattractive, as they may lead to large
errors or (for iterative solvers) to slow convergence.
"""

# ╔═╡ bcccff96-e24f-11ed-2c1c-c33e8f41c539
let
    # Demonstrate the solve with a moderate penalty
    μ = 1e-4
    xhat = (H+A*A'/μ)\(d+A*b[1]/μ)
    
    xx = range(-3, 3, length=100)
    p = plot(xx, xx, (x,y) -> ϕ1([x; y]), st=:contour, legend=false)
    plot!(xx, 1.0 .- xx, linewidth=2)
    plot!([u_ns[1]], [u_ns[2]], markercolor=:white, marker=true)
    plot!([xhat[1]], [xhat[2]], marker=true)

md"""
Error at μ=$μ: $(norm(xhat-x_ns))

$p
"""
end

# ╔═╡ bcccffa2-e24f-11ed-3f06-7f048dda1127
let
    # Vary penalty to illustrate issues -- uniform improvement
    # with smaller penalty until ill-conditioning kills us
    function penalty_err(μ)
	xhat = (H+A*A'/μ)\(d+A*b[1]/μ)
        norm(xhat-x_ns)
    end
    μs = 10.0.^-(1:14)
    errs = penalty_err.(μs)
    plot(μs, errs, xscale=:log10, yscale=:log10, legend=false)
end

# ╔═╡ bccd00d8-e24f-11ed-1020-974d964f0cdf
md"""
### Lagrange multipliers and KKT systems

The KKT conditions for our equality-constrained problem say that the
gradient of

$$L(x,\lambda) = \phi(x) + \lambda^T (A^T x-b)$$

should be zero. In matrix form, the KKT system (saddle point system)

$$\begin{bmatrix}
    H & A \\
    A^T & 0
  \end{bmatrix}
  \begin{bmatrix} x \\ \lambda \end{bmatrix} =
  \begin{bmatrix} d \\ b \end{bmatrix}.$$

If $A$ and $H$ are well-conditioned, then so is this system, so there
is no bad numerical behavior. The system also retains whatever
sparsity was present in the original system matrices $H$ and
$A$. However, adding the Lagrange multipliers not only increases the
number of variables, but the extended system lacks any positive
definiteness that $H$ may have.

When there are relatively few constraints and a fast solver with $H$
is available, an attractive way to solve this KKT system is the so-called
range-space method, which we recognize as just block Gaussian elimination:

$$\begin{align*}
  A^T H^{-1} A \lambda &= A^T H^{-1} d - b \\
  x &= H^{-1} (d - A\lambda).
\end{align*}$$

Rewritten as we might implement it, we have

$$\begin{align*}
  H x_0 &= d \\
  H Y   &= A \\
  (A^T Y) \lambda &= A^T x_0 - b \\
  x &= x_0 - Y \lambda.
\end{align*}$$


The KKT system is closely related to the penalty formulation that we saw
in the previous subsection, in that if we use Gaussian elimination to
remove the variable $\lambda$ in 

$$\begin{bmatrix}
    H & A \\
    A^T & -\mu I
  \end{bmatrix}
  \begin{bmatrix} \hat{x} \\ \lambda \end{bmatrix} =
  \begin{bmatrix} d \\ b \end{bmatrix},$$

we have the Schur complement system 

$$(H+\mu^{-1} AA^T) \hat{x} = d + \mu^{-1} A b,$$

which is identical to the stationary point condition for the
quadratically penalized objective.
"""

# ╔═╡ bccd00e0-e24f-11ed-0a93-29a54a079f04
xλ = [H A; A' 0.0] \ [d; b]

# ╔═╡ bccd00f4-e24f-11ed-3a8c-99d6d8e637bf
x_ns-xλ[1:end-1]  # Error

# ╔═╡ bccd00f4-e24f-11ed-05a2-61cead4ce5f4
H*xλ[1:end-1]-d  # ∇ϕ(x)

# ╔═╡ bccd00fe-e24f-11ed-3da5-0917bdf014c9
xλ[end]*A  # λ ∇c(x)

# ╔═╡ bccd011c-e24f-11ed-1907-5b7a9ca40ff6
md"""
Note that the constrained stationarity condition is

$$\nabla \phi(x_*) + \lambda \nabla c(x_*) = 0,$$

and we can use this to estimate the Lagrange multipliers from an
approximation via a penalty method.
"""

# ╔═╡ bccd0130-e24f-11ed-115e-27c944f26199
let
    μ = 1e-4
    xhat = (H+A*A'/μ)\(d+A*b[1]/μ)
    r = H*xhat-d
    md"λ ≈ $(-norm(r)^2/(A'*r))"
end

# ╔═╡ bccd01b2-e24f-11ed-0798-bdd7762f446c
md"""
### Uzawa iteration

Block Gaussian elimination is an attractive approach when we have a
fast solver for $H$ and there are not too many constraints.  When
there are a relatively large number of constraints, we might seek an
alternate method.  One such method is the *Uzawa iteration*

$$\begin{align*}
  H x_{k+1} &= d - A \lambda_k \\
  \lambda_{k+1} &= \lambda_{k} + \omega (A^T x_{k+1}-b)
\end{align*}$$

where $\omega > 0$ is a relaxation parameter.  We can eliminate
$x_{k+1}$ to get the iteration

$$\lambda_{k+1} 
  = \lambda_k + \omega (A^T H^{-1} (d-A\lambda_k) - b)
  = (I-\omega A^T H^{-1} A) \lambda_k + \omega (A^T H^{-1} d  - b),$$

which is a Richardson iteration on the Schur complement equation 
$(A^T H^{-1} A) \lambda = A^T H^{-1} d - b$.  We can precondition and
accelerate the Uzawa iteration in a variety of ways, as you might
guess from our earlier discussion of iterative methods for solving
linear systems.
"""

# ╔═╡ bccd028e-e24f-11ed-3311-05bb123417ed
md"""
### Augmenting the Lagrangian

From a solver perspective, the block 2-by-2 structure of the KKT system
looks highly attractive. Alas, we do *not* require that $H$ be
positive definite, nor even that it be nonsingular; to have a unique
global minimum, we only need positive definiteness of the projection of
$H$ onto the null space (i.e. $Q_2^T H Q_2$ should be positive
definite). This means we cannot assume that (for example) $H$ will admit
a Cholesky factorization.

The augmented Lagrangian approach can be seen as solving the constrained
system

$$\mbox{minimize } \frac{1}{2} x^T H x - d^T x + \frac{1}{2\mu} \|A^T x-b\|^2
  \mbox{ s.t. } A^T x = b.$$

The term penalizing nonzero $\|A^T x-b\|$ is, of course, irrelevant at
points satisfying the constraint $A^T x = b$. Hence, the constrained
minimum for this augmented objective is identical to the constrained
minimum of the original objective.  However, if the KKT conditions for
the modified objective take the form

$$\begin{bmatrix}
    H+\mu^{-1}AA^T & A \\
    A^T & 0
  \end{bmatrix}
  \begin{bmatrix} x \\ \lambda \end{bmatrix} =
  \begin{bmatrix} d + \mu^{-1} A b \\ b \end{bmatrix}.$$


Now we do not necessarily need to drive $\mu$ to zero to obtain a good
solution; but by choosing $\mu$ small enough, we can ensure that $H +
\mu^{-1} AA^T$ is positive definite (assuming that the problem is
convex subject to the constraint).  This can be helpful when we want
to use a Cholesky factorization or a method like CG, but the original
$H$ matrix is indefinite or singular.
"""

# ╔═╡ bccd02a0-e24f-11ed-3b79-49fd07106f87
begin
    # Set up a non-positive-definite problem for linearly-constrained QP
    # (2D so that we can plot)
    H2 = [4.0  1.0 ;
	  1.0  -1.0 ]
    
    ϕ2(xy) = xy'*H2*xy/2 - xy'*d
    c2(xy) = A'*xy - b[1]
    ϕ3(xy) = ϕ2(xy) + 3*norm(c2(xy))^2
    xy_sol = [H2 A; A' 0.0] \ [d; b]
end

# ╔═╡ bccd02ca-e24f-11ed-38d8-9f6b8818d0d9
let
    xx = range(-3, 3, length=100)
    plot(xx, xx, (x,y) -> ϕ2([x; y]), st=:contour, legend=false)
    plot!(xx, xx, (x,y) -> ϕ3([x; y]), st=:contour, linestyle=:dash)
    plot!(xx, 1.0 .- xx, linewidth=2)
    plot!([xy_sol[1]], [xy_sol[2]], marker=true)
end

# ╔═╡ bccd02d2-e24f-11ed-0290-e11840cb9cf5
let
    # Sanity check: do we have a constrained min (vs max or saddle)?
    xx = range(-3, 3, length=100)
    plot(xx, [ϕ2([x; 1.0-x]) for x in xx], legend=false)
    plot!(xx, [ϕ3([x; 1.0-x]) for x in xx], linestyle=:dash)
end

# ╔═╡ bccd02de-e24f-11ed-1607-d305442f1667
# Find the solution via method of Lagrange multipliers
[H2 A; A' 0.0] \ [d; b]

# ╔═╡ bccd02e8-e24f-11ed-2447-57f4427162fd
# Augment the Lagrangian so the (1,1) submatrix is positive definite
let
    σ = 2.0
    [H2+A*A'/σ A; A' 0.0] \ [d+A*b[1]/σ; b]
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[compat]
IterativeSolvers = "~0.9.2"
LinearMaps = "~3.10.0"
Plots = "~1.38.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "09b2aa7ba68ef0934769b96b7a00265d547a00ac"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "011a22022ed2fb0352a9bded0fa9d3793a8db362"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "7ea8ead860c85b27e83d198ea54bb2f387db9fc3"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.1+1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "37e4657cd56b11abe3d10cd4a1ec5fbdb4180263"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.7.4"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ee342fcc2b8762c43a60dfbbf73bc2258703af19"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.19"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["ChainRulesCore", "LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "4af48c3585177561e9f0d24eb9619ad3abf77cc7"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.10.0"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "5b3e170ea0724f1e3ed6018c5b006c190f80e87d"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.5"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "5434b0ee344eaf2854de251f326df8720f6a7b55"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.10"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "0b829474fed270a4b0ab07117dce9b9a2fa7581a"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.12"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─bcccf69a-e24f-11ed-14a8-85b0acb3b500
# ╠═bcccf780-e24f-11ed-0075-7d1896fdc11f
# ╠═bcccf7b2-e24f-11ed-244e-bf7a0eb3c9d2
# ╠═bcccf7bc-e24f-11ed-1b76-29e34e1bd44d
# ╠═bcccf7c4-e24f-11ed-3c04-bd6d278d695c
# ╟─bcccf956-e24f-11ed-1f7c-0925bcb8773c
# ╟─bcccfb06-e24f-11ed-1ed3-e9762b9d18f8
# ╟─bcccfb90-e24f-11ed-1559-390ce7575c03
# ╠═bcccfb98-e24f-11ed-1281-3b2d11300cfb
# ╠═bcccfbae-e24f-11ed-3de0-817dbc26f386
# ╟─bcccfc6e-e24f-11ed-3ea1-5d397d767757
# ╠═bcccfc76-e24f-11ed-0404-bb8cc740f1cd
# ╠═bcccfc80-e24f-11ed-0e99-6f4f4a9d0920
# ╟─bcccfd0c-e24f-11ed-1b3c-7dd8ce4dd14b
# ╟─bcccfe4c-e24f-11ed-218d-ad5aaa45060c
# ╠═bcccfe56-e24f-11ed-2580-45a4e0e55a94
# ╟─bcccff8c-e24f-11ed-0118-ed93112b26d7
# ╠═bcccff96-e24f-11ed-2c1c-c33e8f41c539
# ╠═bcccffa2-e24f-11ed-3f06-7f048dda1127
# ╟─bccd00d8-e24f-11ed-1020-974d964f0cdf
# ╠═bccd00e0-e24f-11ed-0a93-29a54a079f04
# ╠═bccd00f4-e24f-11ed-3a8c-99d6d8e637bf
# ╠═bccd00f4-e24f-11ed-05a2-61cead4ce5f4
# ╠═bccd00fe-e24f-11ed-3da5-0917bdf014c9
# ╟─bccd011c-e24f-11ed-1907-5b7a9ca40ff6
# ╠═bccd0130-e24f-11ed-115e-27c944f26199
# ╟─bccd01b2-e24f-11ed-0798-bdd7762f446c
# ╟─bccd028e-e24f-11ed-3311-05bb123417ed
# ╠═bccd02a0-e24f-11ed-3b79-49fd07106f87
# ╠═bccd02ca-e24f-11ed-38d8-9f6b8818d0d9
# ╠═bccd02d2-e24f-11ed-0290-e11840cb9cf5
# ╠═bccd02de-e24f-11ed-1607-d305442f1667
# ╠═bccd02e8-e24f-11ed-2447-57f4427162fd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
