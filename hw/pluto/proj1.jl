### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 47491e45-793a-4d76-9844-6ee8541b39ec
using LinearAlgebra

# ╔═╡ ddcd8bf7-701e-4ae0-a99a-70300f378b46
using Plots

# ╔═╡ 4b3ae49a-8e99-49bc-8df5-4c0edac0c6c7
using SparseArrays

# ╔═╡ 61c04f45-6940-4901-9623-c8a45263cdd9
using SuiteSparse

# ╔═╡ 7d10a635-0c09-4bad-9098-fe2f2281192f
using SpecialFunctions

# ╔═╡ 216b8e9c-f238-4b3c-b406-0479194bef7b
using QuadGK

# ╔═╡ d5ef0b1a-a8a8-11ed-145e-0dd7a3b820b1
md"""
# Proj 1: Approximation with RBFs

A classic function approximation scheme used for interpolating data at scattered points involves the use of a *radial basis function* (RBF), typically denoted by $\phi$.  For a function $f : \mathbb{R}^d \rightarrow \mathbb{R}$, we approximate $f$ by

$$s(x) = \sum_{i=1}^n \phi(\|x-x_i\|_2) c_i$$

where the points $\{x_i\}_{i=1}^n$ are known as *centers* -- we will assume for the purpose of this assignment that these are all distinct.  We sometimes write this more concisely as

$$s(x) = \Phi_{xX} c$$

where $\Phi_{xX}$ is a row vector with entries $(\Phi_{xX})_j = \phi(\|x-x_i\|_2)$.

The coefficient vector $c$ may be chosen by interpolating at the centers; that is, we write $s(x_i) = f(x_i),$ or more compactly

$$\Phi_{XX} c = f_X$$

where $\Phi_{XX} \in \mathbb{R}^{n \times n}$ is the matrix with entries $(\Phi_{XX})_{ij} = \phi(\|x_i-x_j\|)$ and $f_X \in \mathbb{R}^n$ is the vector with entries $(f_X)_i = f(x_i)$.  When $\phi$ is a *positive definite* RBF, the matrix $\Phi_{XX}$ is guaranteed to be positive definite.

There are many reasons to like RBF approximations.  There is a great deal of theory associated with them, both from a classic approximation theory perspective and from a statistics perspective (where $\phi$ is associated with the covariance of a *Gaussian process*).  But for this project, we also like RBF approximations because they naturally give rise to many different types of numerical linear algebra problems associated with solving linear systems!  We will explore some of these in the current project.
"""

# ╔═╡ aa387a30-4bb1-4d2f-95ba-f8761c190e0c
md"""
## Logistics

You should complete tasks 1 and 2 by Friday, Feb 17; tasks 3-5 are due by Friday, Feb 24.

You are encouraged to work in pairs on this project. I particularly encourage you to try pair-thinking and pair-programming -- you learn less if you just try to partition the problems!  You should produce short report addressing the analysis tasks, and a few short codes that address the computational tasks. You may use any Julia functions you might want.

You are allowed (even encouraged!) to read outside resources that talk about these types of computations -- including [my own notes](https://www.cs.cornell.edu/courses/cs6241/2021sp/lec/2021-04-13.pdf) from my ["Numerical Methods for Data Science" course](https://www.cs.cornell.edu/courses/cs6241/2021sp/).  It is possible that you'll find references that tell you outright how to solve a subproblem; if so, feel free to take advantage of them *with citation*!  You may well end up doing more work to find and understand the relevant resources than you would doing it yourself from scratch, but you will learn interesting things along the way.

Most of the code in this project will be short, but that does not make it easy. You should be able to convince both me and your partner that your code is right. A good way to do this is to test thoroughly. Check residuals, compare cheaper or more expensive ways of computing the same thing, and generally use the computer to make sure you don't commit silly errors in algebra or coding. You will also want to make sure that you satisfy the efficiency constraints stated in the tasks.
"""

# ╔═╡ 805aef68-705f-46ea-915d-6a6a0ae276aa
md"""
## Code setup

We will be using several built-in packages for this project.  We'll use the `LinearAlgebra` and `Plots` packages in all most all of our computational homeworks and projects in this course, but here we also use the `SparseArrays` and `SuiteSparse` packages for dealing with sparse linear algebra.  We also use the `SpecialFunctions` package and `QuadGK` for one of the tasks where we use kernel approximations to estimate integrals.
"""

# ╔═╡ 6d11343a-729a-4ad5-8050-87d7f7b60a75
md"""
### A bestiary of basis functions

We consider several possible radial basis functions, outlined below:

- The *squared exponential* basis function (`ϕ_se`) is the default choice used in many machine learning applications.  It is infinitely differentiable, and decays quickly away from the origin.

- The *inverse multiquadric* function (`ϕ_imq`) is also infinitely differentiable, but decays much more slowly away from the origin.

- The *Matern* family of functions is frequently used in geospatial statistics.  These functions have a smoothness parameter; the Matérn 1/2 function (`ϕ_mat12`) is continuous but not differentiable, Matérn 3/2 (`ϕ_mat32`) is continuously differentiable but not twice differentiable, and Matérn 5/2 (`ϕ_mat52`) is twice continuously differentiable but not three times differentiable.

- The *Wendland* family of functions is compactly supported: that means it is exactly zero far enough away from the origin.  The definition of these functions depends on the dimension $d$ of the underlying space in which we are doing interpolation.  Here we define the 2D version of this function that is twice differentiable (`ϕ_w21`).
"""

# ╔═╡ d608b861-d5a6-4327-a9df-08962596097c
begin
	ϕ_se(r) = exp(-r^2)
	ϕ_imq(r) = 1.0/sqrt(1.0 + r^2)
	ϕ_mat12(r) = exp(-r)
	ϕ_mat32(r) = (1.0+sqrt(3.0)*r)*exp(-sqrt(3.0)*r)
	ϕ_mat52(r) = (1.0+sqrt(5.0)*r+5.0/3.0*r^2)*exp(-sqrt(5.0)*r)
	ϕ_w21(r) = max(1.0-r, 0.0)^4*(4.0*r+1.0)
end

# ╔═╡ 39735dbd-58b1-4b6f-8979-a262ccea1e9e
md"""
#### Derivatives

For some of our computations later, it will also be useful to have the derivatives of the various radial basis functions we work with.
"""

# ╔═╡ 6868be8b-ade4-40cd-bb20-52b5dc995d90
begin
	dϕ_se(r) = -2.0*r*exp(-r^2)
	dϕ_imq(r) = -r/(1.0 + r^2)^1.5
	dϕ_mat12(r) = -exp(-r)
	dϕ_mat32(r) = -3.0*r*exp(-sqrt(3.0)*r)
	dϕ_mat52(r) = -5.0/3.0*r*(1 + sqrt(5.0)*r)*exp(-sqrt(5.0)*r)
	dϕ_w21(r) = -20*r*max(1.0-r, 0.0)^3
end

# ╔═╡ 08090a10-342d-4c74-8e09-504463bf9c54
md"""
I am as prone to calculus errors and coding typos as anyone, so when I code derivatives, I also like to include a *finite difference check*; that is, I compare my supposedly-exact derivative comptutation to the approximation

$$f'(x) \approx \frac{f(x+h)-f(x-h)}{2h},$$

which has an absolute error bounded by $h^2 \max_{|z| \leq h} |f'''(z)|/6$ (which we usually write as $O(h^2)$.)
"""

# ╔═╡ 2b258864-6fb4-4c77-b4de-e8c84219befe
let
	function test_dϕ(tag, ϕ, dϕ)
		r = 0.123
		h = 1e-4
		dϕ_fd = (ϕ(r+h)-ϕ(r-h))/2/h  # Finite difference estimate
		dϕ_an = dϕ(r)                # Analytic computation
		relerr = abs(dϕ_fd-dϕ_an)/abs(dϕ_an)
		"| $tag | $relerr |\n"
	end

Markdown.parse(
"""
| RBF type | Relerr |
|----------|--------|
""" *
test_dϕ("SE",           ϕ_se,    dϕ_se) *
test_dϕ("IMQ",          ϕ_imq,   dϕ_imq) *
test_dϕ("Matérn 1/2",   ϕ_mat12, dϕ_mat12) *
test_dϕ("Matérn 3/2",   ϕ_mat32, dϕ_mat32) *
test_dϕ("Matérn 5/2",   ϕ_mat52, dϕ_mat52) *
test_dϕ("Wendland 2,1", ϕ_w21,   dϕ_w21))
end

# ╔═╡ 29d7aae9-b74c-4068-b338-abb79e4b6dff
md"""
There are a few other things worth noting about our test code, as we get used to Julia programming:

- We put the entire tester inside a `let` block, which defines a local scope for variable names.  The function `test_dϕ` that we defined here cannpt be seen outside this block.

- Our tester `test_d\phi` takes functions as an input argument.

- The last expression in the `let` block is the return value, which in this case is a Markdown cell.  The Julia Markdown support includes support for rendering tables, which we have done here.  We have also used the `Markdown.parse` function directly rather than using the `md` macro, making it a little easier to program the output in this case.  Note that `*` on strings represents concatenation in Julia.
"""

# ╔═╡ fb168c40-8a45-4a1d-b422-a2acadc7080a
md"""
#### Length scales

We often define radial basis functions to have a *length scale* $l$, i.e.

$$\phi(r) = \phi_0(r/l).$$

We left out the length scale in the initial definitions of our RBFs, but we add it in now.
"""

# ╔═╡ dd612e83-4e3f-4818-be74-8521c7184e80
function scale_rbf(ϕ0, dϕ0, l=1.0)
	ϕ(r) = ϕ0(r/l)
	dϕ(r) = dϕ0(r/l)/l
	ϕ, dϕ
end

# ╔═╡ 06f19807-f34c-44a0-973d-a5e9ccae890d
md"""
### RBF matrices and vectors

To compute with our interpolating functions, we need to manipulate matrices $\Phi_{XY}$ with entries $\phi(\|x_i-y_j\|)$.  We represent collections of points as matrices $X = \begin{bmatrix} x_1 & \ldots & x_n \end{bmatrix} \in \mathbb{R}^{d \times n}$.  As is usual in Julia, we use an exclamation mark as part of the name for functions that mutate their inputs.
"""

# ╔═╡ 4e870974-efe5-45af-866a-8295b5521202
function form_ΦXY!(result, ϕ, X :: AbstractMatrix, Y :: AbstractMatrix)
	for j = 1:size(Y)[2]
		for i = 1:size(X)[2]
			result[i,j] = ϕ(norm(X[:,i]-Y[:,j]))
		end
	end
	result
end

# ╔═╡ 10fdcb75-aa48-4126-8c5c-e0cb209aa7fb
md"""
We decorated the definitions in the previous block with `AbstractVector` and `AbstractMatrix` types in order to also have a specialized 1D version (where we use numbers to represent individual points, and vectors for collection of points).
"""

# ╔═╡ 063c3bc7-f5e9-4c05-9773-b15b35419674
function form_ΦXY!(result, ϕ, X :: AbstractVector, Y :: AbstractVector)
	form_ΦXY!(result, reshape(X, 1, length(X)), reshape(Y, 1, length(Y)))
end

# ╔═╡ a381e468-c393-4438-a182-b4aacd6d01a5

function form_ΦXY(ϕ, X :: AbstractMatrix, Y :: AbstractMatrix)
	form_ΦXY!(zeros(size(X)[2], size(Y)[2]), ϕ, X, Y)
end

# ╔═╡ aff78c0e-c4c7-413c-a7a0-818977a6c265
md"""
### RBF matrix factorization

We create an `RBFCenters` structure to keep track of the location of the centers in our approximation scheme along with the associated function of the RBF matrix.
We are going to want to allow the interpolation of multiple functions for the same set of centers (for example), so we keep the representation of the centers and the factorization of $\Phi_{XX}$ separate from the coefficient vector for interpolating any specific function.

The implementation is slightly complicated by us typically keeping some extra storage around so that we can add new centers without reallocating storage.
"""

# ╔═╡ ae9609b9-f379-4b1e-b56b-dbf48f39442c
begin

	# Structure representing a collection of RBF centers and an associated matrix rep
	mutable struct RBFCenters
		ϕ :: Function        # RBF function
		dϕ :: Function       # RBF derivative function
		Nmax :: Integer      # Max points that we can accommodate without expansion
		n :: Integer         # Number of points we are currently maintaining
		Xstore :: Matrix     # Storage for location of centers
		Rstore :: Matrix     # Storage for upper triangular Cholesky factor
	end

	
	# Constructor for centers object
	function RBFCenters(ϕ, dϕ, d :: Integer, Nmax=100)
		Xstore = zeros(d, Nmax)
		Rstore = zeros(Nmax, Nmax)
		RBFCenters(ϕ, dϕ, Nmax, 0, Xstore, Rstore)
	end

	
	# Alternate constructor (takes in initial points)
	function RBFCenters(ϕ, dϕ, X :: AbstractMatrix, Nmax=100)
		d = size(X)[1]
		n = size(X)[2]
		Nmax = max(n, Nmax)
		rbfc = RBFCenters(ϕ, dϕ, d, Nmax)
		add_centers_ref!(rbfc, X)
		rbfc
	end

	
	# Get number of centers
	Base.length(c :: RBFCenters) = c.n


	# Misc getters
	#  rbfc.X     = array of current centers
	#  rbfc.Xiter = iterator over current centers
	#  rbfc.R     = upper triangular part of storage matrix
	#  rbfc.F     = Cholesky factorization object on top of storage
	#  rbfc.d     = dimension of ambient space
	function Base.getproperty(c :: RBFCenters, v :: Symbol)
		if v == :X
			view(c.Xstore, :, 1:c.n)
		elseif v == :Xiter
			eachcol(view(c.Xstore, :, 1:c.n))
		elseif v == :R
			UpperTriangular(view(c.Rstore, 1:c.n, 1:c.n))
		elseif v == :F
			# NB: This *interprets* the storage as a Cholesky factor;
			#     no actual computation or data movement involved
			Cholesky(UpperTriangular(view(c.Rstore, 1:c.n, 1:c.n)))
		elseif v == :d
			size(c.X)[1]
		else
			getfield(c, v)
		end
	end

	
	# Expand the storage
	function grow!(c :: RBFCenters, new_Nmax)
		if c.Nmax >= new_Nmax
			return
		end
		X0 = c.X
		R0 = c.R
		c.Xstore = zeros(c.d, new_Nmax)
		c.Rstore = zeros(new_Nmax, new_Nmax)
		c.X[:,1:c.Nmax] = X0
		c.R[1:c.Nmax, 1:c.Nmax] = R0
		c.Nmax = new_Nmax
	end


	# Add centers (naive version, completely refactors)
	function add_centers_ref!(c :: RBFCenters, X :: AbstractMatrix)

		# Extend storage if needed, copy in new points
		n0 = c.n
		n1 = c.n + size(X)[2]
		grow!(c, n1)
		c.Xstore[:,n0+1:n1] = X
		c.n = n1

		# Rebuild factorization
		R = view(c.Rstore, 1:n1, 1:n1)
		form_ΦXY!(R, c.ϕ, c.X, c.X)
		cholesky!(Hermitian(R))

	end
	
end

# ╔═╡ e5ed88e0-72af-4c14-94ce-9c6df307e617
function form_ΦXY(ϕ, X :: AbstractVector, Y :: AbstractVector)
	form_ΦXY(ϕ, reshape(X, 1, length(X)), reshape(Y, 1, length(Y)))
end

# ╔═╡ 6fd7e279-db90-4bc9-8444-108b513cae57
md"""
Once we have a factorization of the matrix $\Phi_{XX}$ through $n$ centers (which takes $O(n^3)$ time to compute), we can use that factorization to solve $\Phi_{XX} c = f_X$ in $O(n^2)$ time.  We can then evaluate $s(x) = \phi_{xX} c$ at a new point in $O(n)$ time.  The `solve` and `feval` routines compute $c$ and evaluate $s(x)$, respectively.  Note that the use of iterators means that we never actually form $\phi_{xX}$ in memory -- we just compute with it element by element.

We sometimes also want to compute gradients $\nabla s(x)$, which we can do with the `dfeval` function.  If the implementation of `dfeval` seems slightly cryptic, note that the chain rule gives us

$$\nabla_x \phi(\|x-x_j\|_2) = \phi'(\|x-x_j\|_2) \frac{x-x_j}{\|x-x_j\|}$$

and we have to do something special when $x = x_j$ (the answer there is zero when differentiation makes sense at all).  Therefore, by linearity of differentiation,

$$\nabla_x s(x) = \sum_{j=1}^n \phi'(\|x-x_j\|_2) \frac{x-x_j}{\|x-x_j\|} c_j$$
"""

# ╔═╡ a81eef7e-0328-47ac-8b28-889a9acee87a
begin
	# Compute a coefficient vector for a given rhs
	solve(rbfc :: RBFCenters, fX :: AbstractVector) = rbfc.F \ fX
	solve(rbfc :: RBFCenters, f :: Function) = rbfc.F \ [f(x) for x in rbfc.Xiter]

	
	# Evaluate an interpolant at a given point
	function feval(rbfc :: RBFCenters, c :: Vector, x)
		ϕ = rbfc.ϕ
		sum( ϕ(norm(x-xj))*cj for (xj, cj) in zip(rbfc.Xiter, c) )
	end

	
	# Evaluate a gradient at a given point
	function dfeval(rbfc :: RBFCenters, c :: Vector, x)
		dϕ = rbfc.dϕ
		∇ϕ(r) = if norm(r) == 0 0*r else dϕ(norm(r))*r/norm(r) end
		sum( ∇ϕ(x-xj)*cj for (xj, cj) in zip(rbfc.Xiter, c) )
	end
end

# ╔═╡ c5791017-6ebb-4550-b621-1e5a8a0cfaa9
md"""
### Hello world!

Let's use our machinery now for a toy task of 1D interpolation of $\cos(x)$ on the interval from $[0, 2\pi]$.  We will use an equi-spaced sampling mesh, with the squared exponential RBF.
"""

# ╔═╡ e73d3496-8562-4c64-94d3-bb60efe6556d
md"""
## Task 1: Fast mean temperatures

So far, we have discussed approximating one function at many points.  Sometimes, we would like to quickly approximate something about *many* functions.  For example, in 1D, suppose we have streaming temperature measurements $\theta_j(t)$ taken at $n$ fixed points $x_j \in [0, 1]$, from which we could estimate an instantaneous temperature field

$$\hat{\theta}(x, t) = \sum_{j=1}^n \phi(\|x-x_j\|) c_j$$

by the interpolation condition $\hat{\theta}(x_i, t) = \theta_i(t)$.  Assuming we have a factorization for $\Phi_{XX}$, we can compute the estimated mean temperature

$$\tilde{\theta}(t) = \int_0^1 \hat{\theta}(x, t) \, dx = \sum_{j=1}^n \left( \int_0^1 \phi(\|x-x_j\|) \, dx \right) c_j$$

for any given $t$ by solving for the $c$ vector ($O(n^2)$ time) and taking the appropriate linear combination of integrals ($O(n)$ time).  Here it is worth noting that

$$\int_0^1 \phi(\|x-x_j\|) \, dx = \int_0^{x_j} \phi(s) \, ds + \int_{0}^{1-x_j} \phi(s) \, ds$$

and for $\phi(r) = \exp(-r^2/l^2)$, we have

$$\int_0^x \phi(s) \, ds = \frac{l \sqrt{\pi}}{2} \operatorname{erf}(x/l),$$

where the error function $\operatorname{erf}(x)$ is implemented as `erf(x)` in the Julia `SpecialFunction` library.  We implement this scheme in the following code, which runs in $O(n^3) + O(n^2 m)$ time with $n$ sensors and $m$ measurements.
"""

# ╔═╡ 55b23eb8-c8b8-4c51-aa0c-ca7bb193c299
md"""
An example run with about 2000 virtual sensors and 1000 samples takes a bit under 3 seconds to process this way on my laptop.
"""

# ╔═╡ 46738ffe-343c-403d-8cdc-c7cb7fbea75a
md"""
Rewrite this code to take $O(n^3) + O(mn)$ time by reassociating the expression $\tilde{\theta}(t) = w^T \Phi_{XX}^{-1} \theta_X(t)$.  Compare the timing of the example given above to the timing of your routine; do the numbers seem reasonable?  At the same time, do a comparison of the results to make sure that you do not get something different!  You may want to use the testing and timing harness below.
"""

# ╔═╡ 03d9918f-1ab1-4838-b432-b2980b829e54
md"""
## Task 2: Adding data points

The `add_centers_ref!` function adds centers to an existing `RBFCenters` object, then recomputes the factorization from scratch.  If we add $k$ centers to an existing object with $n$ centers, this costs $O((n+k)^3)$.  However, we can compute the same thing in $O(n^2 k + k^3)$ time by extending the existing Cholesky factorization using what we know about block factorization methods.

It may be helpful to use `ldiv!(A, B)`, which overwrites the storage in `B` with `A\B` (where `A` can be a factorization object or a matrix with a simple solver, like an upper triangular or diagonal matrix).
"""

# ╔═╡ 6bbd913d-c420-4a38-a886-4d3e850a358b
function add_centers!(rbfc :: RBFCenters, Xnew :: AbstractMatrix)
	# TODO: Replace this with code that extends the factorization instead of updating.
	#   add_centers_ref!(rbfc, Xnew)
	# We provide a partial solution below.

	add_centers_ref!(rbfc, Xnew)

# Partial code for you to fill in:
#	# Extend storage if needed
#	n0 = rbfc.n
#	n1 = n0 + size(Xnew)[2]
#	grow!(rbfc, n1)
#
#	# Copy in new points
#	rbfc.Xstore[:,n0+1:n1] = Xnew
#
#	# TODO: Extend factorization
#	# End TODO
#
#	# Update the size
#	rbfc.n = n1
#
#	# Return the factorization object
#	rbfc.F
end

# ╔═╡ 92e5faac-bc68-485c-ae45-baa6e175cd7e
function test_add_centers(nsensor, nstart, nbatch)
	l = 1.0/nsensor
	ϕ, dϕ = scale_rbf(ϕ_se, dϕ_se, l)
	
	x_sensor = zeros(1,nsensor)
	x_sensor[:] = range(0, 1, length=nsensor)
	t1 = @time begin
		rbfc1 = RBFCenters(ϕ, dϕ, x_sensor[:,1:nstart])
		for k = nstart:nbatch:nsensor
			add_centers_ref!(rbfc1, x_sensor[:,k+1:min(k+nbatch,nsensor)])
		end
	end
	t2 = @time begin
		rbfc2 = RBFCenters(ϕ, dϕ, x_sensor[:,1:nstart])
		for k = nstart:nbatch:nsensor
			add_centers!(rbfc2, x_sensor[:,k+1:min(k+nbatch,nsensor)])
		end
	end
	relerr = norm(rbfc1.R-rbfc2.R)/norm(rbfc1.R)
	md"Relative error: $relerr"
end

# ╔═╡ 2cc0d3ef-5a41-4851-9c4c-324adf3c412b
test_add_centers(2001, 1500, 50)

# ╔═╡ e5c76888-26b3-4e02-8da4-ca2491192737
md"""
## Task 3: Missing data

Now suppose that we are again dealing with streaming sensor data, but every so often there are entries missing.  If the $k$th measurement is missing, the interpolation conditions for the remaining points can be written in terms of a smaller system of equations where we remove row and column $k$ from the original problem; *or* as

$$\begin{align*}
  f(x_i) &= \sum_{j=1}^n \phi(\|x_i-x_j\|) \hat{c}_j + r_i, & r_i = 0 \mbox{ for } i \neq k  \\
  \hat{c}_k &= 0.
\end{align*}$$

Equivalently, we have

$$\begin{bmatrix}
  \Phi_{XX} & e_k \\
  e_k^T & 0
\end{bmatrix} 
\begin{bmatrix} \hat{c} \\ r_k \end{bmatrix}
=
\begin{bmatrix} \tilde{f}_X \\ 0 \end{bmatrix}$$

where $\tilde{f}_X$ agrees with $f_X$ except in the $k$th element.  If we set $(\tilde{f}_X)_k = 0$, then $-r_k$ is the value at $x_k$ of the interpolant through the remaining data -- potentially a pretty good guess for the true value.

Using block elimination on the system above, complete the following routine to fill in a single missing value in an indicated location.  Your code should take $O(n^2)$ time (it would take $O(n^3)$ to refactor from scratch).
"""

# ╔═╡ abe3c998-fed1-4942-9ac9-690b48484679
# Assuming that entry k of the measurement vector fX is missing,
# fill it with an approximate value from interpolating the remaining points.
#
function fill_missing!(rbfc :: RBFCenters, fX :: AbstractVector, k)
	# TODO: Complete this routine
	fX[k]
end

# ╔═╡ 225973ba-ba0e-4b62-af87-226f3e742a93
md"""
## Task 4: Sparse approximation

So far, we have used dense matrix representations for everything.  As you have seen, this is often basically fine for up to a couple thousand data points, provided that we are careful to re-use factorizations and organize our computations for efficiency.  When we have much more data, though, the $O(n^3)$ cost of an initial factorization gets to be excessive.

There are several ways to use *data sparsity* of the RBF matrix for faster (approximate) solves.  For this project, though, we will focus on a simple one: using ordinary sparsity of $\Phi_{XX}$ for compactly supported kernels like the Wendland, or approximating that sparsity for rapidly-decaying kernels like the squared exponential.

We are going to solve 2D function approximation problems here.  To do this, we will want a couple of additional routines that we will provide.
"""

# ╔═╡ 2411d159-c27f-4c71-999b-e19330aff402
md"""
### Code preliminaries
"""

# ╔═╡ 94854854-d067-4196-8149-a92fbaa58301
md"""
#### Sampling the space

When we want to sample something in more than one spatial dimension and aren't just using a regular mesh, it is tempting to choose random samples.  But taking independent uniform draws is not an especially effective way of covering a space – random numbers tend to clump up. For this reason, [low discrepancy sequences](https://en.wikipedia.org/wiki/Low-discrepancy_sequence) are often a better basis for sampling than (pseudo)random draws. There are many such generators; we use a relatively simple one based on an additive recurrence with a multiplier based on the ["generalized golden ratio"](http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/).
"""

# ╔═╡ 106f5dd5-e973-470b-ab2c-408ac7fd13fc
function kronecker_quasirand(d, N, start=0)
    
    # Compute the recommended constants ("generalized golden ratio")
    ϕ = 1.0+1.0/d
    for k = 1:10
        gϕ = ϕ^(d+1)-ϕ-1
        dgϕ= (d+1)*ϕ^d-1
        ϕ -= gϕ/dgϕ
    end
    αs = [mod(1.0/ϕ^j, 1.0) for j=1:d]
    
    # Compute the quasi-random sequence
    Z = zeros(d, N)
    for j = 1:N
        for i=1:d
            Z[i,j] = mod(0.5 + (start+j)*αs[i], 1.0)
        end
    end
    
    Z
end

# ╔═╡ 531bf8f8-d947-439f-8455-62543af167cc
md"""
#### Finding neighbors

If we work with a radial basis function $\phi$ that is zero past some distance $r_{\mathrm{cutoff}}$, then $(\Phi_{XX})_{ij} \neq 0$ only if $\|x_i-x_j\| \leq r_{\mathrm{cutoff}}$.  We can, of course, find all such pairs in $O(n^2)$ time by checking all possible pairs.  But things can go faster with a simple data structure.  In 2D, think of putting every point $x_i$ into a square "bucket" with side length $r_{\mathrm{cutoff}}$.  Then all points within $r_{\mathrm{cutoff}}$ of $x_i$ must be in one of nine buckets: the one containing $x_i$, or any of its neighbors.  If we give the buckets identifiers and sort them in "column major" order (and sort their contents accordingly), this can significantly reduce the number of potential neighbors to any point that we might have to check.
"""

# ╔═╡ e6e63df0-abe3-4cdd-924b-807b6b0524f1
# Given a 2-by-n matrix xy of coordinates and a cutoff threshold,
# quickly find all pairs of points within the cutoff from each other.
# Returns
#  - xy_new -- a reordered version of the input matrix in "bucket order"
#  - p -- the reordering to bucket order (xy_new = xy[:,p])
#  - neighbors(q) -> Iq, rq -- a function that takes a query point q and finds
#    indices Iq of points within the cutoff of q and rq of the distances from q
#    to those points.  The index is with respect to the bucket order.
#  - I, J, r -- three parallel arrays such that I[k], J[k], r[k] represents the
#    indices of two points within the cutoff of each other and the distance
#    between them.
#
function find_neighbors2d(xy, rcutoff)

	# Logic for computing integer bucket indices
	n = size(xy)[2]
	xmin = minimum(xy[1,:])
	xmax = maximum(xy[1,:])
	ymin = minimum(xy[2,:])
	ymax = maximum(xy[2,:])
	idx1d(x) = floor(Int, x/rcutoff) + 1
	nx = idx1d(xmax-xmin)
	idx(x,y) = idx1d(x-xmin) + (idx1d(y-ymin)-1)*nx

	# Assign points to buckets, sort by bucket index
	buckets = idx.(xy[1,:], xy[2,:])
	p = sortperm(buckets)
	buckets = buckets[p]
	xy = xy[:,p]

	# Set up three parallel arrays for tracking neighbors closer than the cutoff
	I = zeros(Int, n)
	J = zeros(Int, n)
	r = zeros(Float64, n)
	I[:] = 1:n
	J[:] = 1:n
	function process_edge(i, j)
		rij = norm(xy[:,i]-xy[:,j])
		if rij <= rcutoff
			push!(I, i); push!(J, j); push!(r, rij)
			push!(I, j); push!(J, i); push!(r, rij)
		end
	end

	# Use the bucket structure to check all pairwise interactions quickly
	# (process each edge once, self-loops handled separately).  Note it's
	# fine if we "wrap around" the edge of the bucket array -- it just means
	# checking some extra budgets.
	for i = 1:n
		j0 = searchsortedfirst(buckets, buckets[i]-nx-1)
		j1 = searchsortedlast(buckets, buckets[i]-nx+1)
		for j = j0:j1
			process_edge(i, j)
		end
		j2 = searchsortedfirst(buckets, buckets[i]-1)
		for j = j2:i-1
			process_edge(i, j)
		end
	end

	# Find neighbors of a query point
	function neighbors(qxy)
		b = idx(qxy[1], qxy[2])
		nbrs = Int[]
		rs = Float64[]
		for col = -1:1
			j0 = searchsortedfirst(buckets, b+col*nx-1)
			j1 = searchsortedlast(buckets, b+col*nx+1)
			for j = j0:j1
				rqj = norm(qxy-xy[:,j])
				if rqj <= rcutoff
					push!(nbrs, j)
					push!(rs, rqj)
				end
			end
		end
		nbrs, rs
	end

	# Return permuted coordinates, etc
	xy, p, neighbors, I, J, r
end

# ╔═╡ d5843955-efbc-4693-82cd-4737a2e168af
md"""
#### Sparse RBF centers

Using the data structure in `find_neighbors2d`, we can create a 2D sparse version of the `RBFCenters` data structure.  We will not provide functionality to add new centers after initialization, so we will not bother with making it mutable.
"""

# ╔═╡ 217d0466-893e-40fa-8233-7fffac1fc3ca
begin

	# Structure representing a collection of RBF centers and an associated matrix rep
	struct RBFCenters2Ds
		ϕ :: Function         # RBF function
		dϕ :: Function        # RBF derivative function
		X :: Matrix           # Coordinates
		p :: Vector           # Permutation from original to new index
		ΦXX                   # Sparse RBF matrix
		F                     # Sparse Cholesky factorization of ΦXX
		neighbors :: Function # Function to query neighbors
	end

	function RBFCenters2Ds(ϕ, dϕ, X, rcutoff)
		Xnew, p, neighbors, I, J, r = find_neighbors2d(X, rcutoff)
		ΦXX = sparse(I, J, ϕ.(r))
		F = cholesky(ΦXX)
		RBFCenters2Ds(ϕ, dϕ, Xnew, p, ΦXX, F, neighbors)
	end

end

# ╔═╡ 8506560e-84a8-4193-a3ad-bf17e3c51e93
md"""
As before, we want the ability to solve for a right hand side, evaluate a function, and evaluate a gradient.
"""

# ╔═╡ 6bf0c274-d0e5-4f42-8957-849a054fd3b1
begin
	# Compute a coefficient vector for a given rhs
	solve(rbfc :: RBFCenters2Ds, fX :: AbstractVector) = rbfc.F \ fX
	solve(rbfc :: RBFCenters2Ds, f :: Function) = rbfc.F \ [f(x) for x in eachcol(rbfc.X)]

	
	# Evaluate an interpolant at a given point
	function feval(rbfc :: RBFCenters2Ds, c :: Vector, x)
		ϕ = rbfc.ϕ
		sum( ϕ(rj)*c[j] for (j, rj) in zip(rbfc.neighbors(x)...) )
	end

	
	# Evaluate a gradient at a given point
	function dfeval(rbfc :: RBFCenters2Ds, c :: Vector, x)
		dϕ = rbfc.dϕ
		∇ϕ(r) = if norm(r) == 0 0*r else dϕ(norm(r))*r/norm(r) end
		sum( ∇ϕ(x-rbfc.X[j])*c[j] for j in rbfc.neighbors(x)[1] )
	end
end

# ╔═╡ 0092a9ac-a65e-4ebe-9da1-d407795de8df
let

	# Set up a sampling mesh
	x = zeros(1, 8)
	for j = 0:7
		x[1,j+1] = 2*π*j/7
	end

	# Fit the cosine function
	rbfc = RBFCenters(ϕ_se, dϕ_se, x)
	c = solve(rbfc, (x) -> cos(x[1]))

	# Plot on a finer mesh
	xx = range(0, 2*π, length=100)
	sxx = [feval(rbfc, c, [x]) for x in xx]
	plot(xx, cos.(xx), label="cos(x)")
	plot!(xx, sxx, linestyle=:dash, label="s(x)")
	scatter!(x, cos.(x), markersize=5, label=nothing)

end

# ╔═╡ c9b2c474-2f42-4f98-948a-fcffec7b67df
# Given a set of sensor locations x and a time series of temperature measurements
# Θ[i,j] = measurement of sensor i at time j, return the corresponding time series
# of mean temperature estimates.
#
function mean_temperatures(x, Θ)

	# Set up the RBF and RBF matrix for interpolation
	l = 1.0/length(x)
	ϕ, dϕ = scale_rbf(ϕ_se, dϕ_se, l)
	X = reshape(x, 1, prod(size(x)))
	rbfc = RBFCenters(ϕ, dϕ, X)

	# Computation for weights
	wt(x) = l*sqrt(π)/2 * (erf(x/l) + erf((1.0-x)/l))
	w = [wt(xi) for xi in x]

	# Compute the mean temperature for each column
	θmean = zeros(size(Θ)[2])
	for j = 1:size(Θ)[2]
		c = solve(rbfc, Θ[:,j])
		θmean[j] = dot(w, c)
	end

	θmean
end

# ╔═╡ 82e67a32-6faf-4647-a4d6-a09f0e2f7921
let
	nsensor = 2001
	ntimes = 1000
	
	x_sensor = zeros(1,nsensor)
	x_sensor[:] = range(0, 1, length=nsensor)
	Θ = [cos(x * 2π * j/ntimes) + 10.0*j/ntimes for x=x_sensor[:], j = 1:ntimes]

	θmean = mean_temperatures(x_sensor, Θ)
	plot(θmean)
end

# ╔═╡ 87a3e81e-c5a8-441d-9e16-73af92a6b932
# Given a set of sensor locations x and a time series of temperature measurements
# Θ[i,j] = measurement of sensor i at time j, return the corresponding time series
# of mean temperature estimates.
#
function mean_temperatures2(x, Θ)

	# Set up the RBF and RBF matrix for interpolation
	l = 1.0/length(x)
	ϕ, dϕ = scale_rbf(ϕ_se, dϕ_se, l)
	X = reshape(x, 1, prod(size(x)))
	rbfc = RBFCenters(ϕ, dϕ, X)

	# TODO: Write your fast implementation here (replace the line below)
	mean_temperatures(x, Θ)
end

# ╔═╡ 7fed8d83-55b2-4562-8059-0920b9ae4d42
function test_sensor_demo(nsensor, ntimes)
	x_sensor = zeros(1,nsensor)
	x_sensor[:] = range(0, 1, length=nsensor)
	Θ = [cos(x * 2π * j/ntimes) + 10.0*j/ntimes for x=x_sensor[:], j = 1:ntimes]

	@time θmean1 = mean_temperatures(x_sensor, Θ)
	@time θmean2 = mean_temperatures2(x_sensor, Θ)
	norm(θmean1-θmean2, Inf)
end

# ╔═╡ 9d41a94d-ddb7-4da8-8f6f-468558447d38
test_sensor_demo(2001, 1000)

# ╔═╡ 1bbf7800-79ee-4584-8ece-e65a4613e4a9
let

	# Set up a sampling mesh
	x = zeros(1, 15)
	x[:] = range(0.0, 2π, 15)
	fX = cos.(x[:])

	# Fit the cosine function
	rbfc = RBFCenters(ϕ_se, dϕ_se, x)
	c = solve(rbfc, fX)

	# Try filling in a missing point 4
	fXbad = copy(fX)
	fXbad[4] = 0.0
	fx4_interp1 = fill_missing!(rbfc, fXbad, 4)

	# Compare to the obvious-but-expensive algorithm
	mask = ones(Bool, 15)
	mask[4] = false
	rbfc2 = RBFCenters(ϕ_se, dϕ_se, x[:, mask])
	c2 = solve(rbfc2, fX[mask])
	fx4_interp2 = feval(rbfc2, c2, x[:,4])

	relerr1 = abs(fXbad[4] - fx4_interp2)/abs(fx4_interp2)
	err2 = abs(fx4_interp1 - cos(x[1,4]))

md"""
- `fill_missing!` vs reference rel diff: $relerr1
- Abs error in approximation: $err2
"""
end

# ╔═╡ 6cefd91a-e682-4f21-873e-cfe8261d3fb9
md"""
#### Hello world

As before, we do a "hello world" test to illustrate the behavior of the solver.  In this case, we approximate a simple function from 10K sample points in 2D.
"""

# ╔═╡ 09cabea3-5c3e-4ebe-83f7-d893e5f0bd2e
function test_hello_world_2d(npoints=10000)
	l = 5.0/sqrt(npoints)
	ϕ, dϕ = scale_rbf(ϕ_w21, dϕ_w21, l)
	xy = kronecker_quasirand(2, npoints)
	ftest(x) = cos(x[1])*exp(x[2])

	@time rbfc = RBFCenters2Ds(ϕ, dϕ, xy, l)
	@time c = solve(rbfc, ftest)
	@time ftest([0.5; 0.5]), feval(rbfc, c, [0.5; 0.5])
end

# ╔═╡ 43ef840a-e38c-4a73-820a-8aae19a35acf
test_hello_world_2d()

# ╔═╡ 936749a8-beb3-4367-befd-12b430fe0dbf
md"""
### The questions

For the case of the Wendland kernel, the cutoff is exact, and $\Phi_{XX}$ is truly sparse.  However, while the squared exponential kernel is never exactly zero, it does get very small.  Therefore, we might try to do the same thing here.
"""

# ╔═╡ e8df269f-c51b-4c22-8c0f-0b3b42d66c73
function hello_world_2d_se(l=0.02, σ=4.0)
	l = 0.02 # Length scale for kernel
	σ = 4.0  # Number of length scales to extend out
	
	ϕ, dϕ = scale_rbf(ϕ_se, dϕ_se, l)
	xy = kronecker_quasirand(2, 10000)
	ftest(x) = cos(x[1])*exp(x[2])

	@time rbfc = RBFCenters2Ds(ϕ, dϕ, xy, σ*l)
	@time fX = [ftest(x) for x in eachcol(rbfc.X)]
	@time c = solve(rbfc, fX)

	rbfc, c, fX, ftest
end

# ╔═╡ bdcf7793-7c3b-4cfa-a8e0-4fbffa8a4684
let
	rbfc, c, fX, ftest = hello_world_2d_se(0.02, 4.0)
	smid = ftest([0.5; 0.5])
	fmid = feval(rbfc, c, [0.5; 0.5])
	smid-fmid, fmid
end

# ╔═╡ 4d17a930-57d5-4301-b0c2-3c127cd2e87b
md"""
Even a little playing around with this code should illustrate that it's delicate. 
 Notice that we have changed the length scale to be even shorter.  For example, try running with $l = 0.05$; what happens?
"""

# ╔═╡ f02e55dc-480b-4626-93f5-2807b8a23463
md"""
*Answer*: TODO
"""

# ╔═╡ 955fc65f-01fb-4cfe-b161-dbc098c16ff0
md"""
The code doesn't crash with $l = 0.02$, and our sanity check computation on a simple test function looks pretty good.  But we would like to better understand the error in this approximation.  Let $\hat{s}(x)$ be the sparsified approximator that we have computed above, i.e.

$$\hat{s}(x) = \sum_{\|x-x_i\| \leq r_{\mathrm{cutoff}}} \phi(\|x-x_i\|) c_{0,i}$$

Let us start by showing that

$$|\hat{s}(x)-s(x)| \leq \|c_0\|_1 \exp(-\sigma^2) + \|c-c_0\|_1.$$

*Hint/sketch*: Write $\hat{\Phi}_{xX}$ as the vector of evaluations with cutoff, so that $\hat{s}(x) = \hat{\Phi}_{xX} c_0$ and $s(x) = \Phi_{xX} c$.  Add and subtract $\Phi_{xX} c_0$; apply the triangle inequality; use the fact that $|u \cdot v| \leq \|u\|_1 \|v\|_\infty$ for any $u$ and $v$; and derive simple bounds on $\|\Phi_{xX}\|_\infty$ and $\|\hat{\Phi}_{xX}-\Phi_{xX}\|_\infty$.
"""

# ╔═╡ d32ddf29-0440-411a-85ae-5bd23d2cf4e9
md"""
*Answer*: TODO
"""

# ╔═╡ a3a1ea35-88d5-4c5c-a66e-fae9e713e5bd
md"""
The bound is pessimistic, but it gives a sense of what can go wrong: if there is a large error in our coefficients, or if the coefficient norm times the magnitude of the neglected RBF evaluations is big, then we may have large differences between $s$ and $\hat{s}$.

The error in the coefficient vector $\|c-c_0\|_1$ with our initial parameters can be quite large, enough to make our error bounds terrible (even if the error might not be as bad).  But even if the factorization of $\hat{\Phi}_{XX}$ is not good enough to get a very accurate $c$ alone, it is good enough to make progress.  We therefore try an *iterative refinement loop*, combining our approximate solver based on $\hat{\Phi}_XX$ and a matrix vector product with a (still sparse) $\tilde{\Phi}_XX$ that provides a much more approximation to $\Phi_{xX}$.  Your goal: carry out the iterative refinement and give evidence of its convergence.
"""

# ╔═╡ b2a75033-03b6-4632-94ef-49b88fdb9d30
function hello_world_2d_se_itref(l=0.02, σ=4.0, σ1=6.0)

	# Set up problem for which we will do a factorization
	rbfc, c0, fX, ftest = hello_world_2d_se(l, σ)

	# Form matvec for reference ΦXX (with larger cutoff σ1)
	Xnew, p, neighbors, I, J, r = find_neighbors2d(rbfc.X, σ1*l)
	ΦXX = sparse(I, J, rbfc.ϕ.(r))
	function mulΦXX(y, result)
		result[p] = ΦXX*y[p]
		result
	end
	mulΦXX(y) = mulΦXX(y, zeros(size(ΦXX)[1]))

	c = c0
	# TODO: Iterative refinement loop (take maybe 10-12 steps)

	c
end

# ╔═╡ eba24972-69cb-45db-9c69-79331572befe
md"""
We can take a rough guess at the magnitude of the error in $c$ by the magnitude of the corrections taken during the iterative refinement loop.  As for the first term in our error bound, we can compute that easily in Julia.
"""

# ╔═╡ 1d105240-b696-4021-8478-3c8abe3fc896
function hello_world2d_iterf_err_term(l=0.02, σ=4.0, σ1=6.0)
	norm(hello_world_2d_se_itref(l, σ, σ1), 1)*exp(-σ^2)
end

# ╔═╡ ebbf6e0d-6992-438d-adfb-e0ef1d3b0809
hello_world2d_iterf_err_term(0.02, 4.0, 6.0)

# ╔═╡ 2963b657-731c-4358-b18f-a2fc6a7f3979
hello_world2d_iterf_err_term(0.02, 5.0, 6.0)

# ╔═╡ 1950044e-6559-49f9-a6ba-f2e251dc9da2
md"""
## Task 5: Fun with Fekete

The *Fekete points* for RBF interpolation on some domain $\Omega$ are the points that maximize $\det \Phi_{XX}$ -- though it will be convenient for us to instead consider $\log \det \Phi_{XX}$.  In order to maximize such a quantity, we would like to be able to compute derivatives; that is, we seek to compute the matrix of $G$ with entries

$$G_{ij} = \frac{\partial (\log \det \Phi_{XX})}{\partial x_{ij}}$$

where $x_{ij}$ represents component $i$ of the $j$th center.  Some matrix calculus that we will not go into now lets us write this as

$$G_{ij} = \operatorname{tr}\left( \Phi_{XX}^{-1} \, \frac{\partial \Phi_{XX}}{\partial x_{ij}} \right),$$

and we can write the second matrix as

$$\frac{\partial \Phi_{XX}}{\partial x_{ij}} =
  e_j (v^{ij})^T + v^{ij} e_j^T \mbox{ where }
  v^{ij}_k = 
  \begin{cases}
    \phi'(\|x_j-x_k\|) \frac{x_{ij}-x_{ik}}{\|x_j-x_k\|}, & k \neq j \\
    0, & k = j.
  \end{cases}$$

A standard property of traces (the cyclic property) tells us that in general for any matrix $A$ and vectors $u$ and $w$ such that the dimensions make sense, 

$$\operatorname{tr}(Auw^T) = w^T A u$$

Therefore, using the $v$ vectors introduced above, we can write

$$G_{ij} = 2 e_j^T \Phi_{XX}^{-1} v^{ij} = 2 (v^{ij})^T \Phi_{XX}^{-1} e_j$$

For this problem, you should complete the following routines to compute the log determinant of $\Phi_{XX}$ and the derivative matrix $G$ in the indicated time bounds.  A tester is provided below.
"""

# ╔═╡ fa62fb47-d021-4eac-bdc3-40d9aac1ad2f
# TODO: Complete this function (should be O(n) time)
function logdet_ΦXX(rbfc :: RBFCenters)
	1.0
end

# ╔═╡ eff8fba9-45ba-46f8-9b8e-0824e66a3b13
# TODO: Complete this function (should be O(n^3 + n^2 d) time)
function dlogdet_ΦXX(rbfc :: RBFCenters)
	G = zeros(rbfc.d, rbfc.n)
	G
end

# ╔═╡ 94cc426e-6484-41ea-bbc3-4f8fb122e9e2
# Test the analytical formula for a directional derivative versus finite differences
function test_logdet()

	X = rand(2,10)
	δX = rand(2,10)
	h = 1e-5

	# Compute using the analytical formula
	rbfc = RBFCenters(ϕ_se, dϕ_se, X)
	G = dlogdet_ΦXX(rbfc)
	δld_an = dot(G, δX)

	# Compute using finite differences
	logdet_p = logdet_ΦXX(RBFCenters(ϕ_se, dϕ_se, X+h*δX))
	logdet_m = logdet_ΦXX(RBFCenters(ϕ_se, dϕ_se, X-h*δX))
	δld_fd = (logdet_p-logdet_m)/2/h

	abs( (δld_fd-δld_an)/δld_an )
end

# ╔═╡ b549410a-2134-41f6-aae5-124159b15023
test_logdet()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
SuiteSparse = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[compat]
Plots = "~1.38.4"
QuadGK = "~2.8.1"
SpecialFunctions = "~2.1.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "a34a1ee3dac888cd276295c1511edc0c15833c4c"

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
git-tree-sha1 = "844b061c104c408b24537482469400af6075aae4"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.5"

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
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

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
git-tree-sha1 = "9e23bd6bb3eb4300cb567bdf63e2c14e5d2ffdbc"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.5"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "aa23c9f9b7c0ba6baeabe966ea1c7d2c7487ef90"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.5+0"

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
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

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
git-tree-sha1 = "2422f47b34d4b127720a18f86fa7b1aa2e141f29"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.18"

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

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "680e733c3a0a9cea9e935c8c2184aea6a63fa0b5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.21"

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
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

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
git-tree-sha1 = "6503b77492fd7fcb9379bf73cd31035670e3c509"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.3"

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
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "18f84637e00b72ba6769034a4b50d79ee40c84a9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.5"

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
git-tree-sha1 = "87036ff7d1277aa624ce4d211ddd8720116f80bf"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.4"

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

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "786efa36b7eff813723c4849c90456609cf06661"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.1"

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
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

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
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

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
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

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
# ╟─d5ef0b1a-a8a8-11ed-145e-0dd7a3b820b1
# ╟─aa387a30-4bb1-4d2f-95ba-f8761c190e0c
# ╟─805aef68-705f-46ea-915d-6a6a0ae276aa
# ╠═47491e45-793a-4d76-9844-6ee8541b39ec
# ╠═ddcd8bf7-701e-4ae0-a99a-70300f378b46
# ╠═4b3ae49a-8e99-49bc-8df5-4c0edac0c6c7
# ╠═61c04f45-6940-4901-9623-c8a45263cdd9
# ╠═7d10a635-0c09-4bad-9098-fe2f2281192f
# ╠═216b8e9c-f238-4b3c-b406-0479194bef7b
# ╟─6d11343a-729a-4ad5-8050-87d7f7b60a75
# ╠═d608b861-d5a6-4327-a9df-08962596097c
# ╟─39735dbd-58b1-4b6f-8979-a262ccea1e9e
# ╠═6868be8b-ade4-40cd-bb20-52b5dc995d90
# ╟─08090a10-342d-4c74-8e09-504463bf9c54
# ╠═2b258864-6fb4-4c77-b4de-e8c84219befe
# ╟─29d7aae9-b74c-4068-b338-abb79e4b6dff
# ╟─fb168c40-8a45-4a1d-b422-a2acadc7080a
# ╠═dd612e83-4e3f-4818-be74-8521c7184e80
# ╟─06f19807-f34c-44a0-973d-a5e9ccae890d
# ╠═4e870974-efe5-45af-866a-8295b5521202
# ╠═a381e468-c393-4438-a182-b4aacd6d01a5
# ╟─10fdcb75-aa48-4126-8c5c-e0cb209aa7fb
# ╠═063c3bc7-f5e9-4c05-9773-b15b35419674
# ╠═e5ed88e0-72af-4c14-94ce-9c6df307e617
# ╟─aff78c0e-c4c7-413c-a7a0-818977a6c265
# ╠═ae9609b9-f379-4b1e-b56b-dbf48f39442c
# ╟─6fd7e279-db90-4bc9-8444-108b513cae57
# ╠═a81eef7e-0328-47ac-8b28-889a9acee87a
# ╟─c5791017-6ebb-4550-b621-1e5a8a0cfaa9
# ╠═0092a9ac-a65e-4ebe-9da1-d407795de8df
# ╟─e73d3496-8562-4c64-94d3-bb60efe6556d
# ╠═c9b2c474-2f42-4f98-948a-fcffec7b67df
# ╟─55b23eb8-c8b8-4c51-aa0c-ca7bb193c299
# ╠═82e67a32-6faf-4647-a4d6-a09f0e2f7921
# ╟─46738ffe-343c-403d-8cdc-c7cb7fbea75a
# ╠═87a3e81e-c5a8-441d-9e16-73af92a6b932
# ╠═7fed8d83-55b2-4562-8059-0920b9ae4d42
# ╠═9d41a94d-ddb7-4da8-8f6f-468558447d38
# ╟─03d9918f-1ab1-4838-b432-b2980b829e54
# ╠═6bbd913d-c420-4a38-a886-4d3e850a358b
# ╠═92e5faac-bc68-485c-ae45-baa6e175cd7e
# ╠═2cc0d3ef-5a41-4851-9c4c-324adf3c412b
# ╟─e5c76888-26b3-4e02-8da4-ca2491192737
# ╠═abe3c998-fed1-4942-9ac9-690b48484679
# ╠═1bbf7800-79ee-4584-8ece-e65a4613e4a9
# ╟─225973ba-ba0e-4b62-af87-226f3e742a93
# ╟─2411d159-c27f-4c71-999b-e19330aff402
# ╟─94854854-d067-4196-8149-a92fbaa58301
# ╠═106f5dd5-e973-470b-ab2c-408ac7fd13fc
# ╟─531bf8f8-d947-439f-8455-62543af167cc
# ╠═e6e63df0-abe3-4cdd-924b-807b6b0524f1
# ╟─d5843955-efbc-4693-82cd-4737a2e168af
# ╠═217d0466-893e-40fa-8233-7fffac1fc3ca
# ╟─8506560e-84a8-4193-a3ad-bf17e3c51e93
# ╠═6bf0c274-d0e5-4f42-8957-849a054fd3b1
# ╟─6cefd91a-e682-4f21-873e-cfe8261d3fb9
# ╠═09cabea3-5c3e-4ebe-83f7-d893e5f0bd2e
# ╠═43ef840a-e38c-4a73-820a-8aae19a35acf
# ╟─936749a8-beb3-4367-befd-12b430fe0dbf
# ╠═e8df269f-c51b-4c22-8c0f-0b3b42d66c73
# ╠═bdcf7793-7c3b-4cfa-a8e0-4fbffa8a4684
# ╟─4d17a930-57d5-4301-b0c2-3c127cd2e87b
# ╟─f02e55dc-480b-4626-93f5-2807b8a23463
# ╟─955fc65f-01fb-4cfe-b161-dbc098c16ff0
# ╟─d32ddf29-0440-411a-85ae-5bd23d2cf4e9
# ╟─a3a1ea35-88d5-4c5c-a66e-fae9e713e5bd
# ╠═b2a75033-03b6-4632-94ef-49b88fdb9d30
# ╟─eba24972-69cb-45db-9c69-79331572befe
# ╠═1d105240-b696-4021-8478-3c8abe3fc896
# ╠═ebbf6e0d-6992-438d-adfb-e0ef1d3b0809
# ╠═2963b657-731c-4358-b18f-a2fc6a7f3979
# ╟─1950044e-6559-49f9-a6ba-f2e251dc9da2
# ╠═fa62fb47-d021-4eac-bdc3-40d9aac1ad2f
# ╠═eff8fba9-45ba-46f8-9b8e-0824e66a3b13
# ╠═94cc426e-6484-41ea-bbc3-4f8fb122e9e2
# ╠═b549410a-2134-41f6-aae5-124159b15023
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
