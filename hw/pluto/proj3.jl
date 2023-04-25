### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 37922d67-8509-43c0-ba25-49ef4b07c74c
using LinearAlgebra

# ╔═╡ 89628c8b-ba90-4770-a14b-e5791489bd96
using Plots

# ╔═╡ 2b0c526e-9cf1-4cfa-a0dc-2b284eb6260c
using PlutoUI

# ╔═╡ 8f6ee3ca-9d12-4649-8b92-3f4a1810657a
using Roots

# ╔═╡ 93922272-2156-4efa-a3aa-7434397d985b
md"""
# Project 3: Mapping the Unknown
"""

# ╔═╡ 7a672894-5cac-4554-863c-26c2dc261cfa
md"""
## Introduction

The [Taylor-Chirikov standard map](https://en.wikipedia.org/wiki/Standard_map) is a standard example from the study of nonlinear dynamics.  The mapping takes a point $(x, p)$ to a new point $(x', p')$ by the rule

$$\begin{align*}
  p' &= p + K \sin(x) \\
  x' &= x + p'
\end{align*}$$

The variables $x$ and $p$ represent angles, and we will by default choose the representation where $x \in [0, 2\pi)$ and $p \in [-\pi, \pi]$.  As the parameter $K \in \mathbb{R}$, the dynamics of repeatedly applying the standard map get increasingly complicated.
"""

# ╔═╡ 73566ff8-e9e3-4808-bb7c-66c8ca23a504
periodize0(θ) = mod(θ+π, 2π)-π

# ╔═╡ abd4ed45-5c80-4a3b-8965-86266ffbb7a9
periodize1(θ) = mod(θ, 2π)

# ╔═╡ 5b1ed90c-378b-4dc3-8e5d-18a29aeeece9
function chirikov(x, p, K)
	p += K*sin(x)
	x += p
	periodize1(x), periodize0(p)
end

# ╔═╡ 73fba5af-42e8-4a80-9bdb-0f778652eedc
md"""
One way to visualize the dynamics of repeatedly applying a map like the standard map is to draw a *Poincaré plot*: starting from a regular array of points (we choose $x = \pi$ and $p$ ranging from $-\pi$ to $\pi$), we repeatedly apply the map and put a dot at each new point.  We use a slightly bolded dot for each of the starting points, and color the different trajectories differently to make it easier to see the structure.
"""

# ╔═╡ 3ffaee2b-1907-416c-a120-df25e090ee4c
md"""
We can use the [`PlutoUI`](https://github.com/JuliaPluto/PlutoUI.jl) package to build an interactive visualization of the Poincaré plot over a range of parameters.  For small values of $K$, the dynamics all follow low-dimensional invariant sets.  As $K$ increases, we start to see regions of chaos; and once $K$ is large enough, most of the structure dissolves into a sea of chaos.
"""

# ╔═╡ 5abc9dab-3459-4f47-a552-d7f73d72fe5b
@bind Kslider Slider(0:100)

# ╔═╡ 6fdbca9a-2816-4cd0-87bd-9787e07e0a2d
md"""
The goal of this project is to understand how to compute different types of invariant sets and (to the extent we can) how those invariant sets change behavior as $K$ changes.
"""

# ╔═╡ e0a1e895-9e46-4c2b-ad7d-526397357741
md"""
## Warm up

Let's denote a general point in the phase space as $z = (x, p) \in \mathbb{T}^2$, where $\mathbb{T} = \mathbb{R} / 2\pi\mathbb{Z}$ is a torus (i.e., $x$ and $p$ are angles and are only defined up to integer multiples of $2\pi$).  We will also write $G : \mathbb{T}^2 \rightarrow \mathbb{T}^2$ to denote the standard map.
"""

# ╔═╡ 5e0816e0-595e-49ff-9c29-d0f2e75f6e5c
md"""
### Computing the Jacobian

For most of what follows, we will need the Jacobian $G'(z)$.

**Question**: Complete the code below to fill it in.
"""

# ╔═╡ 3074e08a-4e05-4002-a340-45257ca25eed
function chirikov(xp, K)
	x, p = chirikov(xp[1], xp[2], K)
	[x; p]
end

# ╔═╡ ab99286e-5344-4987-96c0-471c0d3d3f2d
function poincare_chirikov(K; xstart=π*ones(50), pstart=range(-π, π, length=50))
	npt = 500
	nstart = length(xstart)

	xs = zeros(npt, nstart)
	ps = zeros(npt, nstart)

	for j = 1:nstart
		x = xstart[j]
		p = pstart[j]
		xs[1,j] = x
		ps[1,j] = p
		for i = 2:npt
			x, p = chirikov(x, p, K)
			xs[i,j] = x
			ps[i,j] = p
		end
	end

	p = scatter(xs[1,:], ps[1,:], markersize=2, markerstrokewidth=0, legend=false, title="K = $K")
	for j = 1:nstart
		scatter!(xs[:,j], ps[:,j], markersize=1, markerstrokewidth=0)
	end
	p
end

# ╔═╡ b985a8b7-715e-4191-875c-b33c612535a4
poincare_chirikov(Kslider/20)

# ╔═╡ 545d9e11-c1a7-43b5-9970-290c2f9400ce
function Jchirikov(xp, K)
	# TODO: Fill this in with the right thing!
	[1.0 0.0; 0.0 1.0]
end

# ╔═╡ ab7b9f47-ccdd-4af2-b6b5-15c57aaa4578
md"""
As usual, we do a finite difference check to make sure that the Jacobian is implemented correctly.
"""

# ╔═╡ 33a8b824-a0a7-45e8-90ad-988a77be8254
let
	xp = rand(2)
	u = rand(2)
	h = 1e-6
	K = 0.6
	∂c_∂u = Jchirikov(xp, K)*u
	∂c_∂u_fd = (chirikov(xp+h*u, K)-chirikov(xp-h*u, K))/(2*h)
	norm(∂c_∂u-∂c_∂u_fd)/norm(∂c_∂u)
end

# ╔═╡ dfa240df-7930-4b89-9f3e-1645c3182c35
md"""
The Jacobian should have the property that $\det(G'(z)) = 1$ for all $z$.  Geometrically, this means it preserves area.  This is a particular example of a *symplectic* map.
"""

# ╔═╡ 3d30e431-7055-4429-841f-52cb485e1ba5
md"""
### XOXO

For any $K$, the standard map has two fixed points at $(0, 0)$ and $(\pi, 0)$.  If $G(z_*) = z_*$ is a fixed point of $G$, then for starting points $z_0$ near $z_*$, the dynamics of the iteration $z_{k+1} = G(z_k)$ are locally approximated by $\delta z_{k+1} = G'(z_*) \delta z_k$.  Therefore, we classify the fixed point by the eigenvalues of $G'(z_*)$, similar to the way we do phase plane analysis of fixed points of differential equations.  Because $\det(G'(z_*)) = 1$, the possible classifications are:

- The Jacobian has complex conjugate eigenvalues on the unit circle.  In this case, $z_*$ is a *center* (also called an "O point").  Near an O point, the linearized dynamics trace out *quasiperiodic orbits*.
- The Jacobian has real eigenvalues, with one greater than one in magnitude and the other less than one in magnitude.  In this case, $z_*$ is a *saddle* (also called an "X point").
- The Jacobian has a double eigenvalue at $1$ or $-1$.

Because the characteristic polynomial of a 2-by-2 matrix $A$ is

$$p(z) = z^2 - \operatorname{tr}(A) z + \det(A),$$

and we know $\det(G'(z_*)) = 1$, we can classify the fixed points based purely on the trace of the Jacobian.

**Questions**: For what values of the trace do we have a center or a saddle?  If you play with the visualization, you can observe that for a critical value of $K$ the fixed point at $(\pi, 0)$ changes type; what is that critical value of $K$?
"""

# ╔═╡ 1b1acf3b-7dd7-4c86-8bf2-aa65ecce6ea8
function check_trace(J)
	t = tr(J)
	# TODO: Fill in to output "circle", "saddle", or "boundary" along with the trace
	ptype = "bunny"  # Replace this
end

# ╔═╡ c3228699-6f4f-47e9-a550-c5340907fc39
md"""
### Periodic points

A *periodic point* with period $q$ is a point $z_0$ such that iterating $z_{k+1} = G(z_k)$ for $q$ steps yields $z_q = z_0.$  At $K = 0$, the periodic points of period $q$ have the form $(x, 2\pi r/q)$, where $r$ is integer and $x$ can be anything.  When $p$ is not a rational multiple of $2\pi$, the iterates fill out all possible $x$ (modulo $2\pi$) and $x$ remains constant; in this case, we say the invariant set with constant $p$ is a *quasi-periodic* orbit.

While the periodic points and quasi-periodic orbits are more complicated to analyze for $K > 0$, $(0,\pi)$ is a periodic point with period 2, since $(0,\pi)$ maps to $(\pi, \pi)$ and vice-versa.

Let $G_{(q)}(z_*)$ denote the $q$-fold iteration of $G$ starting from $z_*$.  For points $z_0 = z_*$, we have $z_{k+1} = G(z_k)$ leading to $z_q = z_0$.  For starting points close to $z_0$, we have the *variational equation*

$$\delta z_{k+1} = G'(z_k) \delta z_k$$

and so

$$G_{(q)}'(z_*) \delta z_0 = G'(z_{q-1}) G'(z_{q-2}) \ldots G'(z_0) \delta z_0.$$

This is really just an application of the chain rule, but it is a useful one.

If we apply a cyclic permutation of the Jacobians, we will get a different matrix but one with the same eigenvalues; for example, $G'(z_2) G'(z_1) G'(z_0)$ and $G'(z_1) G'(z_0) G'(z_2)$ have the same eigenvalues.

**Question**: Explain -- why is this?

Just as we classify fixed points as centers or saddles according to the eigenvalues of $G'$, we can classify periodic points as centers or saddles according to the eigenvalues of $G_{(q)}'$.  In the case of periodic points that are centers, we trace out quasiperiodic orbits that hop from one center to the next, drawing circles around each; these structures are sometimes known as "island chains."

**Question**: Carry out the same analysis for the periodic point $(0, \pi)$ that we carried out above for the fixed point $(\pi, 0)$.  For what values of $K$ is this a center, and for what values is it a saddle?
"""

# ╔═╡ 282322e7-b0e8-40eb-883e-2a36c6bdd83e
md"""
## Analyzing periodic points

Not all periodic points are as easy to locate as the ones at $(0,0)$, $(\pi, 0)$, $(0, \pi)$ and $(\pi, \pi)$.  In general, we will need to compute them numerically when $K > 0$.  To do this, it is useful to have a function that computes the iterated map $G_{(q)}$ and its derivative.

**Question**: Complete the following code so that it correctly computes the derivative of $G_{(q)}$ as well as computing $G_{(q)}$ itself.
"""

# ╔═╡ 656f266a-ff4e-4450-9b2c-93764e0977b9
function chirikovp(xp, K, q)
	J = [1.0 0.0; 0.0 1.0]
	for j = 1:q
		# TODO: Add the line for the Jacobian
		xp = chirikov(xp, K)
	end
	xp, J
end

# ╔═╡ 23a8fc85-0fdd-43d2-a83d-be2151a4f530
md"""
As the basis for finding periodic points, we also would like to compute the residual $G_{(q)}(z)-z$ and its Jacobian.
"""

# ╔═╡ eb64f085-d74a-4108-973d-83cc78dc3805
function chirikovp_resid(xp, K, q)
	Gq_xp, Jq_xp = chirikovp(xp, K, q)
	Gq_xp-xp, Jq_xp-I
end

# ╔═╡ db5e5ba7-faa4-423b-8b16-8264058bd69b
md"""
### Newton for periodic points

We provide a simple Newton iteration for finding periodic points.  Unfortunately, naive Newton does not converge particularly well unless we have very good initial guesses.  We can improve matters some by adding a line search, though this is not a panacea.
"""

# ╔═╡ 4becd6b6-8ae4-4666-94c3-28072b3ced6c
md"""
**Question**: Add a line search to the Newton iteration below.  Use the resulting computation to try to find a period 3 point for $K = 0.5$ with an initial guess of $(\pi, 2\pi/3)$.  Give a convergence plot to show that we ultimately do get quadratic convergence.  What do you observe about your result?  What happens if you use the initial guess of $(\pi, 2.2)$?
"""

# ╔═╡ 79dda135-29e7-4671-8a8a-887ffb0f4358
function find_chirikov_periodic(xp, K, q; monitor=(xp, rnorm)->nothing)
	η = 1e-3
	F, J = chirikovp_resid(xp, K, q)
	for j = 1:100
		rnorm = norm(F)
		monitor(xp, rnorm)
		if rnorm < 1e-10
			return xp, chirikovp(xp, K, q)[2]
		end

		# TODO: Add line search based on reducing the residual norm by
		#       at least 1e-3 * α at each step
		xp -= J\F
		F, J = chirikovp_resid(xp, K, q)
		
	end
	error("Did not converge!")
end

# ╔═╡ b31e382a-a892-45f2-b7ba-789c896e02d6
let
	rnorms = []
	q = 3
	xp, J = find_chirikov_periodic([π; 2π/3], 0.5, q, 
			monitor=(xp,rnorm)->push!(rnorms, rnorm))
	p = plot(rnorms, yscale=:log10, legend=false, xlabel="\$k\$", ylabel="Resid norm")
md"""
From $(\pi, 2\pi/3)$, converge to ($(xp[1]), $(xp[2])) -- it is a $(check_trace(J)).

$p
"""
end

# ╔═╡ 58dbfe90-5d15-4051-b6e7-540af922c831
let
	rnorms = []
	q = 3
	xp, J = find_chirikov_periodic([π; 2.2], 0.5, q, 
			monitor=(xp,rnorm)->push!(rnorms, rnorm))
	p = plot(rnorms, yscale=:log10, legend=false, xlabel="\$k\$", ylabel="Resid norm")
md"""
From $(\pi, 2\pi/3)$, converge to ($(xp[1]), $(xp[2])) -- it is a $(check_trace(J)).

$p
"""
end

# ╔═╡ 4610a45f-307d-41ac-9347-d698e2a64f47
md"""
### Gauss-Newton for periodic points with $x = \pi$

By inspection of the Poincaré plot, a number of periodic points occur on the line $x = \pi$.  We try to find those points by looking for where $\|G_{(q)}(z)-z\|_2$ is zero along that line.  We will follow the strategy of scanning for local minima on a coarse mesh, then refining.  We note that we will likely find many points this way, so we will return a vector of solutions.
"""

# ╔═╡ aa2ece02-fc5e-40df-b4ba-d0c443ead49d
md"""
**Question**: Add Gauss-Newton refinement to the routine below.  Keep only points that converge to a point close to the starting guess (closer than either of the neighbors) for which we have residual tolerance of less than $10^{-10}$.
"""

# ╔═╡ a2826975-345a-4aff-abe4-c263a5a2f73f
function refine_periodicπ(p, K, q)
	# TODO: Replace by Gauss-Newton refinement (you will want to throw
	#       away any points that do not converge to something close to the
	#       initial point)
	F, J = chirikovp_resid([π; p], K, q)
	# Last output is the flag for if it converged -- correct this!
	return p, norm(F), true
end

# ╔═╡ 56fb08e8-0597-48c0-89fc-5a5ffb2f061a
function scan_periodicπ(K, q)

	# Sample periodically on a grid of length ns
	ns = 100
	ps = range(-π, π, length=ns+1)
	ps = ps[1:end-1]

	# Get residual norms
	rnorms = [norm(chirikovp_resid([π; p], K, q)[1]) for p in ps]

	# Scan for local minima
	pmins = []
	for j = 1:ns
		jm = mod(j-2+ns,ns)+1
		jp = mod(j,ns)+1
		if rnorms[j] < rnorms[jm] && rnorms[j] < rnorms[jp]
			push!(pmins, ps[j])
		end
	end

	pout = []
	rout = []
	for p in pmins
		prefined, rnorm, converged = refine_periodicπ(p, K, q)
		if converged && abs(p-prefined) < 2π/ns
			push!(pout, p)
			push!(rout, rnorm)
		end
	end
	
	pout, rout, ps, rnorms
end

# ╔═╡ 041c6894-53a0-4485-9fea-6efa082c9ec3
let
	K = 0.75
	q = 3
    pout, rout, ps, rnorms = scan_periodicπ(K, q)
	p = plot(ps, rnorms, legend=false)
	scatter!(pout, rout)
	_, J = chirikovp([π; pout[1]], K, q)
md"""
First point is a $(check_trace(J))

$p
"""
end

# ╔═╡ 121426fe-92b8-45ea-a794-2576f09b3cab
md"""
### Continuation and computing bifurcations

One of the things we can do with our computation is to plot the behavior of the equilibrium point and of the Jacobian trace as a function of $K$.  To avoid issues with converging to the "wrong" solution, we use do a simple parameter continuation loop with the trivial predictor.
"""

# ╔═╡ 2810efeb-11b6-4967-a925-d0e44c52a0dc
md"""
**Question**: Complete the function below so that it calls `detect` with an accurately resolved $K$ value when a bifurcation occurs (i.e. when the periodic point changes from a circle to a saddle or vice-versa).  You may use whatever zero finding algorithm you want, but make sure that $K$ is good to at least a few digits (more than you get just by taking the closest sample $K$ from the continuation).
"""

# ╔═╡ 04f3cf11-312e-48b3-beff-c9eac15ec120
function trace_periodicπ(pstart, Kstart, Kend, q; nsteps=100, detect=(K)->nothing)

	Ks = range(Kstart, Kend, length=nsteps)
	ps = zeros(nsteps)
	trJ = zeros(nsteps)

	p = pstart
	for j = 1:nsteps
		K = Ks[j]
		p, rnorm, converged = refine_periodicπ(p, K, q)
		if !converged
			error("Did not converge at K=$K")
		end
		ps[j] = p
		_, J = chirikovp([π; p], K, q)
		trJ[j] = tr(J)

		# TODO: If there was a change in type between step j-1 and j,
		#       figure out where it occurred and call detect at that value
		#       of K
	end

	Ks, ps, trJ
end

# ╔═╡ eb04a402-9576-42df-8686-ba38c25cdb7e
let
	q = 3
	Kcrit = []
    Ks, ps, trJ = trace_periodicπ(2.2, 0.5, 2.0, q, detect=(K)->push!(Kcrit, K))
	p = plot(Ks, trJ, legend=false, xlabel="K", ylabel="tr(J)")
	if !isempty(Kcrit)
		vline!(Kcrit)
	end
md"""
Detected bifurcation at K=$(Kcrit)

$p
"""
end

# ╔═╡ f5252810-de88-4aae-afa5-b8028c8b4b77
md"""
## Quasi-periodic orbits

Now we consider the *quasi-periodic* orbits in the neighborhood of $z_* = (\pi, 0)$. 
 When $K$ is not too big, if we start at a point $z_0 = z_* + \Delta_0$ for $\Delta_0$, we expect to see

$$\Delta_{k+1} \approx G'(z_*) \Delta_k$$

where $G'(z_*)$ is similar to a rotation matrix.  The $\Delta_k$ iterates therefore remain approximately on an invariant ellipse (which we typically call an invariant circle, even if it is not quite circular).  This is true only in the linear approximation; but for many small radii, we see that the $\Delta_k$ exactly map out a quasi-periodic orbit, though that orbit is only approximately singular.  This is a consequence of the celebrated [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) (named after Kolmogorov, Arnold, and Moser).

A quasi-periodic orbit is described by a periodic closed curve $z : \mathbb{T} \rightarrow \mathbb{R}^2$ (where $\theta \in \mathbb{R} / 2\pi \mathbb{Z}$ is an angle variable) such that

$$G(z(\theta)) = z(\theta + \tau)$$

Our goal is to (approximately) solve this equation numerically.
"""

# ╔═╡ 61e6a6c9-d85f-4611-aad5-83142bf4e3ff
md"""
### Approximating closed curves

To approximate smooth closed curves will work with truncated Fourier expansion approximations, i.e.

$$z(\theta) = a_0 + \sum_{m=1}^M A_m \begin{bmatrix} \cos(m\theta) \\ \sin(m\theta) \end{bmatrix}$$

where $a_0 \in \mathbb{R}^2$ and $A_m \in \mathbb{R}^{2 \times 2}$.  We are going to store the collection of coefficients in a single vector of size $4M+2$. From this representation, we want to be able to evaluate $z(\theta)$ and $z'(\theta)$.

**Note**: The Fourier series representation is not unique; it can be reparameterized by the transformation $\pi \mapsto \pi + \psi$ for some arbitrary angle $\psi$, yielding a different set of coefficients that describe the same curve.  To get around this non-uniqueness, we will generally want to constraint $A_1$.
"""

# ╔═╡ 4e05dcaa-e8bb-4731-828e-7d098015d179
begin
	# Data structure representing a truncated Fourier series
	struct Fourier2D
	    a :: Vector{Float64}
	    Fourier2D(M::Integer) = new(zeros(4*M+2))
	    Fourier2D(a::Vector) = new(a)
	end

	# Make a copy of the Fourier series expansion object
	Base.copy(z::Fourier2D) = Fourier2D(copy(z.a))

	# Number of terms in the series
	Mterms(z::Fourier2D) = (length(z.a)-2) ÷ 4

	# Access the series coefficients
	a0(z::Fourier2D) = view(z.a, 1:2)
	Am(z::Fourier2D, m) = reshape(view(z.a, 4*m-1:4*m+2), 2, 2)

	# Evaluate the z(θ) at a given angle θ
	function eval(z::Fourier2D, θ)
	    M = Mterms(z)
	    zθ = a0(z)
	    for m = 1:M
	        zθ += Am(z, m) * [cos(m*θ), sin(m*θ)]
	    end
	    return zθ
	end

	# Evaluate z'(θ) at a given angle θ
	function deval(z::Fourier2D, θ)
	    M = Mterms(z)
	    dzθ = zeros(2)
	    for m = 1:M
	        dzθ += Am(z, m) * [-m*sin(m*θ), m*cos(m*θ)]
	    end
	    return dzθ
	end

	# Convenience function to let us write the eval as z(θ)
	(z::Fourier2D)(θ) = eval(z, θ)
end

# ╔═╡ a0b21588-b627-4efc-ac60-0dc7c5e0ccb6
md"""
### The residual

Let $\rho(\theta) = G(z(\theta)) - z(\theta+\tau)$ be the residual at a given $\theta$.  We will seek to minimize

$$\frac{\pi}{N+1} \sum_{j=0}^N \|\rho(\theta_j)\|^2 \approx \frac{1}{2} \int_0^{2\pi} \|\rho(\theta)\|^2 \, d\theta$$

where $\theta_j = 2\pi j/(N+1)$.  We write this as $\pi/(N+1) \|R\|^2$ where

$$R = \begin{bmatrix} \rho(\theta_0) \\ \rho(\theta_1) \\ \vdots \\ \rho(\theta_N) \end{bmatrix}.$$

The following functions evaluate $R$ as a function of $z$ and $\tau$.
"""

# ╔═╡ adc75509-a356-47c9-806e-b4fa18c88bdd
function residv(z, τ, K, N)
    M = Mterms(z)
	R = zeros(2,N+1)
	for j=0:N
		θj = 2π*j/(N+1)
		R[:,j+1] = chirikov(z(θj), K) - z(θj+τ)
	end
	R[:]
end

# ╔═╡ c04054a2-5c35-4690-bd60-4b7b36c9a08b
md"""
**Question**: We will also need the Jacobian matrix consisting of derivatives of the parameters describing $z$ and of $\tau$.  Complete the following code to compute this Jacobian.
"""

# ╔═╡ 68697be9-fcce-4d17-91ee-2c730495d83d
function Jresidv(z, τ, K, N)
    M = Mterms(z)
	J = zeros(2*(N+1), 4*M+3)
	for j=0:N
		θj = 2π*j/(N+1)
		jj = 2*j+1:2*j+2
		Jj = Jchirikov(z(θj), K)
		
		# TODO: Fill in an appropriate row of J
		# - First two columns: derivatives with respect to the $a_0$ term
		# - Next 4M columns: derivatives with respect to each $A_m$
		#   (listing components in column major order)
		# - Last column: derivative with respect to τ

	end
	J
end

# ╔═╡ 63a6f010-4d5e-4010-8f72-666fcb871686
let
	K = 0.6
	N = 100

	z = Fourier2D(10)
	a0(z) .= [π; 0]
	Am(z,1) .= [0.5 0.0; 0.0 0.5]
	z.a[7:end] = 0.3*rand(4*Mterms(z)-4)

	dz = Fourier2D(10)
	dz.a[:] = rand(4*Mterms(dz)+2)

	τ = 0.3
	dτ = 1.0

	h = 1e-6
	zp = Fourier2D(z.a + h*dz.a)
	zm = Fourier2D(z.a - h*dz.a)

	Ju_fd = (residv(zp, τ+h*dτ, K, N)-residv(zm, τ-h*dτ, K, N))/(2*h)
	Ju = Jresidv(z, τ, K, N)*[dz.a; dτ]

	norm(Ju_fd-Ju)/norm(Ju)
end

# ╔═╡ 74fc4b6f-feb0-4248-9df3-8bef75ef603b
md"""
### Initial guess

Let $J = G'(z_*)$ be the Jacobian at the fixed point.  When $z_*$ is a circle point, we can write the Jacobian as

$$J \begin{bmatrix} u & v \end{bmatrix} =
  \begin{bmatrix} u & v \end{bmatrix}
  \begin{bmatrix} c & s \\ -s & c \end{bmatrix}$$

where the eigenvalues of $J$ are $c \pm i s$ and the corresponding eigenvectors are $u \pm i v$.  Near $z_*$, the approximate invariant set is traced out by $u \cos(\theta) + v \sin(\theta)$, and the shift $\tau$ is the angle of the rotation with cosine $c$ and sine $s$.  We will form a specific approximate invariant set.

If we want this parameterization to be positively oriented, we should choose the eigenvector $u + iv$ such that $\det(\begin{bmatrix} u & v \end{bmatrix}) > 0$. We will also choose to normalize so that $z(0) = e_1$; that is, we write

$$A_1 = \frac{1}{r} \begin{bmatrix} u & v \end{bmatrix} Q$$

where the Givens rotation $Q$ satisfies

$$Q^T \begin{bmatrix} u_2 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ r \end{bmatrix}.$$
"""

# ╔═╡ d188c7de-39dd-4e39-b5b5-9e18655f8296
function circle_linear(J)
    eigenJ = eigen(J)
    
    # Choose eigenvector for positive orientation
    if det([real(eigenJ.vectors[:,1]) imag(eigenJ.vectors[:,1])]) > 0
        v = eigenJ.vectors[:,1]
        λ = eigenJ.values[1]
    else
        v = eigenJ.vectors[:,2]
        λ = eigenJ.values[2]
    end
    
    # Rotate and scale so z(0) = e1
    ur = real(v)
    ui = imag(v)
    r = sqrt(ur[2]^2 + ui[2]^2)
    c = ui[2]/r
    s = -ur[2]/r
    A1 = [ur ui] * [c -s ; s c]
    A1[2,1] = 0.0
    A1 /= A1[1,1]
    
    # Return rotation angle and normalized A1 coefficient matrix
    τ = atan(-imag(λ), real(λ))
    return τ, A1
end

# ╔═╡ 4b83e7b2-99f2-49af-b376-5b9ff64622f6
md"""
Using this approximation, we write a function that constructs approximate invariant circles around the fixed point for the standard map.
"""

# ╔═╡ 77f50aef-eb23-4c87-b6ef-f37fabb5b4a7
function circle_chirikov(K, M=10)
	J = Jchirikov([π; 0.0], K)
	z = Fourier2D(M)
	τ, A1 = circle_linear(J)
	a0(z) .= [π; 0]
	Am(z,1) .= A1
	z, τ
end

# ╔═╡ 64f8dcd7-f945-4442-bd29-91d28a8a214e
md"""
We illustrate by overlaying an (approximate) invariant circle on top of the Poincaré plot at $K = 0.6$.
"""

# ╔═╡ ba4fedeb-e15f-454b-8232-79678924081a
let
	K = 0.6
	N = 100

	p = poincare_chirikov(K)
	
	z, τ = circle_chirikov(K)
	Am(z,1) .*= 0.4
	ρnorm2 = sqrt(π/(N+1))*norm(residv(z, τ, K, N))
	
	θs = range(0, 2π, length=100)
	zs = [z(θ) for θ in θs]
	plot!([z[1] for z in zs], [z[2] for z in zs], linecolor=:black, linewidth=2)

md"""
Residual: $(ρnorm2)

$p
"""
end

# ╔═╡ bbe5d751-4db9-4b6f-a1f3-d783c54e788e
md"""
### Gauss-Newton refinement

Our goal now is, given the initial guess at $z$ and $\tau$, refine those parameters in order to minimize $\|R(z, \tau)\|^2$.  We use the Gauss-Newton iteration to accomplish this, stopping when the residual tolerance $\sqrt{\pi/(N+1)} \|R\|_2$ is less than some tolerance.  We have to take care of one additional issue, however: the parameterization for a fixed curve is not determined, and we usually have a one-parameter family of continuous circles!  Therefore, we need to add two equations to pin down the solution.  The simplest way to do this is to fix the point $z(0)$ to whatever the initial value was.  Because we expect to be able to get the residual very close to zero, it is fine to incorporate these constraints as rows in the least squares problem, i.e. minimizing

$$\frac{\pi}{N+1} \|R\|^2 + \|z(0)-z_0\|^2$$

To do this, it is useful to be able to write $z(0)$ as $C a$ where $a$ is the vector of coefficients in the Fourier series; we provide the function to compute $C$ below.
"""

# ╔═╡ b05f2582-592c-45f0-afa3-8ad7daf6c757
function coeffs_z0(z)
	M = Mterms(z)
	C = zeros(2, 4*M+2)
	C[1,1] = 1.0
	C[2,2] = 1.0
	for m = 1:M
		C[1,4*m-1] = 1.0
		C[2,4*m] = 1.0
	end
	C
end

# ╔═╡ 1a885d9a-7058-4c90-bc87-5c086aae9156
md"""
**Question**: Complete the Gauss-Newton refinement code below to update $z$ and $\tau$.  While you should take steps to minimize $π/(N+1) \|R(z, \tau)\|^2 + \|z(0)-z_0\|^2$, you should check convergence based only on $\sqrt{\pi/(N+1)} \|R\|$.
"""

# ╔═╡ eca2fc2d-685d-4525-a1b2-069d90a7413a
function gn_chirikov(z, τ, K, N; rtol=1e-6, monitor=(z,τ,ρnorm)->nothing)
	z = copy(z)
	z0 = z(0)

	# TODO: Replace this with a Gauss-Newton loop to update z and τ
	F = residv(z, τ, K, N)
	ρnorm = sqrt(π/(N+1))*norm(F)
	z, τ, ρnorm
end

# ╔═╡ 2e4dbf6f-32b5-4cf7-b949-0eedf0debbd0
let
	K = 0.6
	N = 100
	
	z, τ = circle_chirikov(K)
	Am(z,1) .*= 0.4
	ρnorm2 = sqrt(π/(N+1))*norm(residv(z, τ, K, N))

	ρhist = []
	z, τ, _ = gn_chirikov(z, τ, K, N, monitor=(z, τ, ρnorm)->push!(ρhist, ρnorm))
	ρnorm2b = sqrt(π/(N+1))*norm(residv(z, τ, K, N))

	#pcvg = plot(ρhist, yscale=:log10, legend=false, 
	#	        ylabel="Residual norm", xlabel="\$k\$")
	
	p = poincare_chirikov(K)
	θs = range(0, 2π, length=100)
	zs = [z(θ) for θ in θs]
	plot!([z[1] for z in zs], [z[2] for z in zs], linecolor=:black, linewidth=2)

md"""
- Residual: $(ρnorm2)
- Refined: $(ρnorm2b)


$p
"""
end

# ╔═╡ 75af3aaf-84e5-480d-a768-c31c9a3613e9
md"""
### Continuation

Finally, we consider the problem of trying to see how far out from the fixed point we can push the invariant circles that we are computing.  We do this by computing incrementally larger circles until we get convergence failure.
"""

# ╔═╡ 4e4c8a70-0f88-4379-90fd-d90f87e45fd6
md"""
**Question**: Explain in words what the code above is doing.  Try running the code above for a few values of $K$ (I recommend $K = 0.6$, $K = 1.7$, $K = 2.5$, at least).  What do you observe about $\tau/2\pi$?  Do you observe anything visually about where the iteration seems to stop?
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"

[compat]
Plots = "~1.38.9"
PlutoUI = "~0.7.50"
Roots = "~2.0.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "f1954e8dee5f1a55b38634504e19cccc3dfaf02b"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

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

[[deps.CommonSolve]]
git-tree-sha1 = "9441451ee712d1aec22edad62db1a9af3dc8d852"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.3"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "89a9db8d28102b094992472d333674bd1a83ce2a"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.1"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "0635807d28a496bb60bc15f465da0107fb29649c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "99e248f643b052a77d2766fe1a16fb32b661afd4"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.0+0"

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

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

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

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

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
git-tree-sha1 = "e9d68fe4b5f78f215aa2f0e6e6dc9e9911d33048"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.4"

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
git-tree-sha1 = "186d38ea29d5c4f238b2d9fe6e1653264101944b"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.9"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

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

[[deps.Roots]]
deps = ["ChainRulesCore", "CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "b45deea4566988994ebb8fb80aa438a295995a6e"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.10"

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

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

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

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

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

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

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
# ╟─93922272-2156-4efa-a3aa-7434397d985b
# ╠═37922d67-8509-43c0-ba25-49ef4b07c74c
# ╠═89628c8b-ba90-4770-a14b-e5791489bd96
# ╠═2b0c526e-9cf1-4cfa-a0dc-2b284eb6260c
# ╠═8f6ee3ca-9d12-4649-8b92-3f4a1810657a
# ╟─7a672894-5cac-4554-863c-26c2dc261cfa
# ╠═73566ff8-e9e3-4808-bb7c-66c8ca23a504
# ╠═abd4ed45-5c80-4a3b-8965-86266ffbb7a9
# ╠═5b1ed90c-378b-4dc3-8e5d-18a29aeeece9
# ╟─73fba5af-42e8-4a80-9bdb-0f778652eedc
# ╠═ab99286e-5344-4987-96c0-471c0d3d3f2d
# ╟─3ffaee2b-1907-416c-a120-df25e090ee4c
# ╠═5abc9dab-3459-4f47-a552-d7f73d72fe5b
# ╠═b985a8b7-715e-4191-875c-b33c612535a4
# ╟─6fdbca9a-2816-4cd0-87bd-9787e07e0a2d
# ╟─e0a1e895-9e46-4c2b-ad7d-526397357741
# ╟─5e0816e0-595e-49ff-9c29-d0f2e75f6e5c
# ╠═3074e08a-4e05-4002-a340-45257ca25eed
# ╠═545d9e11-c1a7-43b5-9970-290c2f9400ce
# ╟─ab7b9f47-ccdd-4af2-b6b5-15c57aaa4578
# ╠═33a8b824-a0a7-45e8-90ad-988a77be8254
# ╟─dfa240df-7930-4b89-9f3e-1645c3182c35
# ╟─3d30e431-7055-4429-841f-52cb485e1ba5
# ╠═1b1acf3b-7dd7-4c86-8bf2-aa65ecce6ea8
# ╟─c3228699-6f4f-47e9-a550-c5340907fc39
# ╟─282322e7-b0e8-40eb-883e-2a36c6bdd83e
# ╠═656f266a-ff4e-4450-9b2c-93764e0977b9
# ╟─23a8fc85-0fdd-43d2-a83d-be2151a4f530
# ╠═eb64f085-d74a-4108-973d-83cc78dc3805
# ╟─db5e5ba7-faa4-423b-8b16-8264058bd69b
# ╟─4becd6b6-8ae4-4666-94c3-28072b3ced6c
# ╠═79dda135-29e7-4671-8a8a-887ffb0f4358
# ╠═b31e382a-a892-45f2-b7ba-789c896e02d6
# ╠═58dbfe90-5d15-4051-b6e7-540af922c831
# ╟─4610a45f-307d-41ac-9347-d698e2a64f47
# ╟─aa2ece02-fc5e-40df-b4ba-d0c443ead49d
# ╠═a2826975-345a-4aff-abe4-c263a5a2f73f
# ╠═56fb08e8-0597-48c0-89fc-5a5ffb2f061a
# ╠═041c6894-53a0-4485-9fea-6efa082c9ec3
# ╟─121426fe-92b8-45ea-a794-2576f09b3cab
# ╟─2810efeb-11b6-4967-a925-d0e44c52a0dc
# ╠═04f3cf11-312e-48b3-beff-c9eac15ec120
# ╠═eb04a402-9576-42df-8686-ba38c25cdb7e
# ╟─f5252810-de88-4aae-afa5-b8028c8b4b77
# ╟─61e6a6c9-d85f-4611-aad5-83142bf4e3ff
# ╠═4e05dcaa-e8bb-4731-828e-7d098015d179
# ╟─a0b21588-b627-4efc-ac60-0dc7c5e0ccb6
# ╠═adc75509-a356-47c9-806e-b4fa18c88bdd
# ╟─c04054a2-5c35-4690-bd60-4b7b36c9a08b
# ╠═68697be9-fcce-4d17-91ee-2c730495d83d
# ╠═63a6f010-4d5e-4010-8f72-666fcb871686
# ╟─74fc4b6f-feb0-4248-9df3-8bef75ef603b
# ╠═d188c7de-39dd-4e39-b5b5-9e18655f8296
# ╟─4b83e7b2-99f2-49af-b376-5b9ff64622f6
# ╠═77f50aef-eb23-4c87-b6ef-f37fabb5b4a7
# ╟─64f8dcd7-f945-4442-bd29-91d28a8a214e
# ╠═ba4fedeb-e15f-454b-8232-79678924081a
# ╟─bbe5d751-4db9-4b6f-a1f3-d783c54e788e
# ╠═b05f2582-592c-45f0-afa3-8ad7daf6c757
# ╟─1a885d9a-7058-4c90-bc87-5c086aae9156
# ╠═eca2fc2d-685d-4525-a1b2-069d90a7413a
# ╠═2e4dbf6f-32b5-4cf7-b949-0eedf0debbd0
# ╟─75af3aaf-84e5-480d-a768-c31c9a3613e9
# ╠═a702563b-4425-401c-8fba-60f859d2dfa8
# ╟─4e4c8a70-0f88-4379-90fd-d90f87e45fd6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
