### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 0e0fc087-fdb7-42a8-8e55-1883faf4b831
using LinearAlgebra

# ╔═╡ d24fb1e0-2606-4f39-b8c2-35a6c6c6319e
using Plots

# ╔═╡ c52a00f8-ae1e-11ec-39af-5504c28148bf
md"""
# Notebook for 2023-03-29
"""

# ╔═╡ 81180297-5e55-4ee0-910f-72be07a18b06
md"""
## Fixed points and contraction mappings

As discussed in previous lectures, many iterations we consider have the form

$$x^{k+1} = G(x^k)$$

where $G : \mathbb{R}^n \rightarrow \mathbb{R}^n$.  We call $G$ a *contraction* on $\Omega$ if it is Lipschitz with constant less than one, i.e.

$$\|G(x)-G(y)\| \leq \alpha \|x-y\|, \quad \alpha < 1.$$

A sufficient (but not necessary) condition for $G$ to be Lipschitz on $\Omega$ is if $G$ is differentiable and $\|G'(x)\| \leq \alpha$ for all $x \in \Omega$.

According to the *contraction mapping theorem* or *Banach fixed point theorem*, when $G$ is a contraction on a closed set $\Omega$ and $G(\Omega) \subset \Omega$, there is a unique fixed point $x^* \in \Omega$ (i.e. a point such that $x^* - G(x^*)$).  If we can express the solution of a nonlinear equation as the fixed point of a contraction mapping, we get two immediate benefits.

First, we know that a solution exists and is unique (at least, it is unique within $\Omega$).  This is a nontrivial advantage, as it is easy to write nonlinear equations that have no solutions, or have continuous families of solutions, without realizing that there is a problem.

Second, we have a numerical method -- albeit a potentially slow one -- for computing the fixed point. We take the fixed point iteration

$$x^{k+1} = G(x^k)$$

started from some $x^0 \in \Omega$, and we subtract the fixed point equation $x^* = G(x^*)$ to get an iteration for $e^k = x^k-x^*$:

$$e^{k+1} = G(x^* + e^k) - G(x^*)$$

Using contractivity, we get

$$\|e^{k+1}\| = \|G(x^* + e^k) - G(x^*)\| \leq \alpha\|e^k\|$$

which implies that $\|e^k\| \leq \alpha^k \|e^0\| \rightarrow 0.$

When error goes down by a factor of $\alpha > 0$ at each step, we say the iteration is *linearly convergent* (or *geometrically convergent*).  The name reflects a semi-logarithmic plot of (log) error versus iteration count; if the errors lie on a straight line, we have linear convergence.  Contractive fixed point iterations converge at least linarly, but may converge more quickly.
"""

# ╔═╡ 0a85f9e5-d964-4c20-993a-2a902c86cd70
md"""
### A toy example

Consider the function $G : \mathbb{R}^2 \rightarrow \mathbb{R}^2$ given by

$$G(x) = \frac{1}{4} \begin{bmatrix} x_1-\cos(x_2) \\ x_2-\sin(x_1) \end{bmatrix}$$

This is a contraction mapping on all of $\mathbb{R}^2$ (why?).  Let's look at $\|x^k-G(x^k)\|$ as a function of $k$, starting from the initial guess $x^0 = 0$.
"""

# ╔═╡ 50492436-7209-4bfc-92f3-557ba0db4ad9
function test_toy_contraction()
	
	G(x) = 0.25*[x[1]-cos(x[2]); x[2]-sin(x[1])]

	# Run the fixed point iteration for 100 steps
	resid_norms = []
	x = zeros(2)
	for k = 1:100
		x = G(x)
		push!(resid_norms, norm(x-G(x), Inf))
	end

	x, resid_norms
end

# ╔═╡ 090659d5-f155-4b67-af45-c823e782ba12
# Plot the residual vs iteration count
let
	x, resid_norms = test_toy_contraction()
	plot(resid_norms[resid_norms .> 0], 
		xlabel="k", ylabel="resid", label="Fixed point residual", yscale=:log10)
end

# ╔═╡ 7985ec7e-29b0-4cbb-a7b5-b0d650badfd1
md"""
#### Questions

1. Show that for the example above $\|G'\|_\infty \leq \frac{1}{2}$ over all of $\mathbb{R}^2$.  This implies that $\|G(x)-G(y)\|_\infty \leq \frac{1}{2} \|x-y\|_\infty$.

2. The mapping $x \mapsto x/2$ is a contraction on $(0, \infty)$, but does not have a fixed point on that interval.  Why does this not contradict the contraction mapping theorem?

3. For $S > 0$, show that the mapping $g(x) = \frac{1}{2} (x + S/x)$ is a contraction on the interval $[\sqrt{S}, \infty)$.  What is the fixed point?  What is the Lipschitz constant?
"""

# ╔═╡ 23e8fd18-41be-46e4-83cc-778cc5e29849
md"""
## Newton's method for nonlinear equations

The idea behind Newton's method is to approximate a nonlinear $f \in C^1$ by linearizations around successive guesses.  We then get the next guess by finding where the linearized approximaton is zero.  That is, we set

$$f(x^{k+1}) \approx f(x^k) + f'(x^k) (x^{k+1}-x^k) = 0,$$

which we can rearrange to

$$x^{k+1} = x^k - f'(x^k)^{-1} f(x^k).$$

Of course, we do not actually want to form an inverse, and to set the stage for later variations on the method, we also write the iteration as

$$\begin{align*}
f'(x^k) p^k &= -f(x^k) \\
x^{k+1} &= x^k + p^k.
\end{align*}$$
"""

# ╔═╡ 0058da6b-fea2-495f-905a-44f13398a98a
md"""
### A toy example

Consider the problem of finding the solutions to the system

$$\begin{align*}
x + 2y &= 2 \\
x^2 + 4y^2 &= 4.
\end{align*}$$

That is, we are looking for the intersection of a straight line and an ellipse.  This is a simple enough problem that we can compute the solution in closed form; there are intersections at $(0, 1)$ and at $(2, 0)$.  Suppose we did not know this, and instead wanted to solve the system by Newton's iteration.  To do this, we need to write the problem as finding the zero of some function

$$f(x,y) = \begin{bmatrix} x + 2y - 2 \\ x^2 + 4y^2 - 4 \end{bmatrix} = 0.$$

We also need the Jacobian $J = f'$:

$$\frac{\partial f}{\partial (x, y)} = 
\begin{bmatrix} 
  \frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} \\
  \frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y}
\end{bmatrix}.$$
"""

# ╔═╡ 662f3ec9-5cf0-4ea5-a6ad-fd9ce8fa8595
function test_toy_newton(x, y)

	# Set up the function and the Jacobian
	f(x) = [x[1] + 2*x[2]-2; 
		    x[1]^2 + 4*x[2]^2 - 4]
	J(x) = [1      2     ;
	        2*x[1] 8*x[2]]

	# Run ten steps of Newton from the initial guess
	x = [x; y]
	fx = f(x)
	resids = zeros(10)
	for k = 1:10
		x -= J(x)\fx
		fx = f(x)
		resids[k] = norm(fx)
	end

	x, resids
end

# ╔═╡ 81e9aaf3-bfa5-4247-a518-a08e3f7d1c34
ttn = test_toy_newton(1, 2)

# ╔═╡ 371c69b8-8d71-4f5d-b131-394e16f4988c
# Plot the residuals on a semilog scale
let
	resids = ttn[2]
	plot(resids[resids .> 0], yscale=:log10, 
		 ylabel="\$k\$", xlabel="\$f(x_k)\$", legend=false)
end

# ╔═╡ 60857857-aeae-4255-af74-64c535b7d66d
md"""
#### Questions

1. Finding an (real) eigenvalue of $A$ can be posed as a nonlinear equation solving problem: we want to find $x$ and $\lambda$ such that $Ax = \lambda x$ and $x^T x = 1$.  Write a Newton iteration for this problem.
"""

# ╔═╡ 99ba3897-4b14-4ef3-894e-48d6ad95a329
md"""
### Superlinear convergence

Suppose $f(x^*) = 0$.  Taylor expansion about $x^k$ gives

$$0 = f(x^*) = f(x^k) + f'(x^k) (x^*-x^k) + r(x^k)$$

where the remainder term $r(x^k)$ is $o(\|x^k-x^*\|) = o(\|e^k\|)$.  Hence

$$x^{k+1} = x^* + f'(x^{k})^{-1} r(x^k)$$

and subtracting $x^*$ from both sides gives

$$e^{k+1} = f'(x^k)^{-1} r(x^) = f'(x^k)^{-1} o(\|e^k\|)$$

If $\|f'(x)^{-1}\|$ is bounded for $x$ near $x^*$ and $x^0$ is close enough, this is sufficient to guarantee *superlinear convergence*.  When we have a stronger condition, such as $f'$ Lipschitz, we get *quadratic convergence*, i.e. $e^{k+1} = O(\|e^k\|^2)$.  Of course, this is all local theory -- we need a good initial guess!
"""

# ╔═╡ 00f65731-867d-421e-b38d-bbe8a6b0ffcd
md"""
## A more complex example

We now consider a more serious example problem, a nonlinear system that comes from a discretized PDE reaction-diffusion model describing (for example) the steady state heat distribution in a medium going an auto-catalytic reaction. The physics is that heat is generated in the medium due to a reaction, with more heat where the temperature is higher. The heat then diffuses through the medium, and the outside edges of the domain are kept at the ambient temperature. The PDE and boundary conditions are

$$\begin{align*}
\frac{d^2 v}{dx^2} + \exp(v) &= 0, \quad x \in (0,1) \\
v(0) = v(1) &= 0.
\end{align*}$$

We discretize the PDE for computer solution by introducing a mesh $x_i = ih$ for $i = 0, \ldots, N+1$ and $h = 1/(N+1)$; the solution is approximated by $v(x_i) \approx v_i$. We set $v_0 = v_{N+1} = 0$; when we refer to $v$ without subscripts, we mean the vector of entries $v_1$ through $v_N$.  This discretization leads to the nonlinear system

$$f_i(v) \equiv \frac{v_{i-1} - 2v_i + v_{i+1}}{h^2} + \exp(v_i) = 0.$$

This system has two solutions; physically, these correspond to stable and unstable steady-state solutions of the time-dependent version of the model.
"""

# ╔═╡ 1a98df17-3fd6-4ccf-8375-60eab7a6556d
#x: 2023-03-29-autocatalytic
function autocatalytic(v)
	N = length(v)
	fv = exp.(v)
	fv -= 2*(N+1)^2*v
	fv[1:N-1] += (N+1)^2*v[2:N  ]
	fv[2:N  ] += (N+1)^2*v[1:N-1]
	fv
end

# ╔═╡ e98e962c-df4e-439f-bfec-611e298a6ea1
md"""
To solve for a Newton step, we need the Jacobian of $f$ with respect to the variables $v_j$.  This is a tridiagonal matrix, which we can write as

$$J(v) = -h^{-2} T_N + \operatorname{diag}(exp(v))$$

where $T_N \in \mathbb{R}^{N \times N}$ is the tridiagonal matrix with $2$ down the main diagonal and $-1$ on the off-diagonals.
"""

# ╔═╡ ee0d40b8-3114-4093-8f57-49cb7ed8b0fc
#x: 2023-03-29-Jautocatalytic
function Jautocatalytic(v)
	N = length(v)
	SymTridiagonal(exp.(v) .- 2*(N+1)^2, (N+1)^2 * ones(N-1))
end

# ╔═╡ cfdbc815-00d9-4fff-a921-217c845fbc82
md"""
For an initial guess, we use $v_i = \alpha x_i(1-x_i)$ for different values of $\alpha$.  For $\alpha = 0$, we converge to the stable solution; for $\alpha = 20$ and $\alpha = 40$, we converge to the unstable solution.  We eventually see quadratic convergence in all cases, but for $\alpha = 40$ there is a longer period before convergence sets in.  For $\alpha = 60$, the method does not converge at all.
"""

# ╔═╡ fb4b9029-b474-4f94-9863-e4649548c076
#x: 2023-03-29-newton_autocatalytic
function newton_autocatalytic(α, N=100, nsteps=50, rtol=1e-8; 
                              monitor=(v, resid)->nothing)
	v_all = [α*x*(1-x) for x in range(0.0, 1.0, length=N+2)]
	v = v_all[2:N+1]
	for step = 1:nsteps
		fv = autocatalytic(v)
		resid = norm(fv)
		monitor(v, resid)
		if resid < rtol
			v_all[2:N+1] = v
			return v_all
		end
		v -= Jautocatalytic(v)\fv
	end
	error("Newton did not converge after $nsteps steps")
end

# ╔═╡ e9dfc16b-7c22-4154-a819-011872ad2184
function newton_autocatalytic_rhist(α, N=100, nsteps=50, rtol=1e-8)
	rhist = []
	monitor(v, resid) = push!(rhist, resid)
	v = newton_autocatalytic(α, N, nsteps, rtol, monitor=monitor)
	v, rhist
end

# ╔═╡ 58932838-0a88-4866-b1a0-bb2e83eb1737
function test_autocatalytic()
	v0, rhist0 = newton_autocatalytic_rhist(0.)
	v20, rhist20 = newton_autocatalytic_rhist(20.)
	v40, rhist40 = newton_autocatalytic_rhist(40.)

	xx = range(0.0, 1.0, length=102)
	p1 = plot(xx, v0, label="\$\\alpha = 0\$")
	plot!(xx, v20, label="\$\\alpha = 20\$")
	plot!(xx, v40, linestyle=:dash, label="\$\\alpha = 40\$")

	p2 = plot(rhist0, yscale=:log10, label="\$\\alpha=0\$")
	plot!(rhist20, label="\$\\alpha = 20\$")
	plot!(rhist40, linestyle=:dash, label="\$\\alpha = 40\$")

	l = @layout [a b]
	plot(p1, p2, layout=l)
end

# ╔═╡ 8e4bc440-e98a-46c2-ba80-7b2016c61587
test_autocatalytic()

# ╔═╡ 3b2c56f0-5725-45b6-bbdc-71e4aaa15e03
md"""
We can derive a Newton-like fixed point iteration from the observation that if $v$ remains modest, the Jacobian is pretty close to $-h^2 T_N$.  This gives us the iteration

$$h^{-2} T_N v^{k+1} = \exp(v^k)$$

Below, we compare the convergence of this fixed point iteration to Newton's method.  The fixed point iteration does converge, but it shows the usual linear convergence, while Newton's method converges quadratically.
"""

# ╔═╡ 84cdc15b-9cdd-4462-a306-e64352df4518
#x: 2023-03-29-fp_autocatalytic
function fp_autocatalytic(α, N=100, nsteps=500, rtol=1e-8;
						  monitor=(v, resid)->nothing)
	v_all = [α*x*(1-x) for x in range(0.0, 1.0, length=N+2)]
	v = v_all[2:N+1]
	TN = SymTridiagonal(2.0*ones(N), -ones(N-1))
	F = ldlt(TN)
	for step = 1:nsteps
		fv = autocatalytic(v)
		resid = norm(fv)
		monitor(v, resid)
		if resid < rtol
			v_all[2:N+1] = v
			return v_all
		end
		v[:] = F\(exp.(v)/(N+1)^2)
	end
	error("Fixed point iteration did not converge after $nsteps steps (α=$α)")
end

# ╔═╡ c584e561-0b45-4274-affe-746fcccaa0e2
function fp_autocatalytic_rhist(α, N=100, nsteps=500, rtol=1e-8)
	rhist = []
	v = fp_autocatalytic(α, N, nsteps, rtol, monitor=(v,r)->push!(rhist, r))
	v, rhist
end

# ╔═╡ 163e3d56-ad19-4256-9830-3eae026cbdc0
function test_autocatalytic_fp()
	v0, rhist0 = newton_autocatalytic_rhist(0.)
	v0f, rhist0f = fp_autocatalytic_rhist(0.)

	xx = range(0.0, 1.0, length=102)
	p1 = plot(xx, v0, label="Newton")
	plot!(xx, v0f, label="Fixed point")

	p2 = plot(rhist0[rhist0.>0], yscale=:log10, label="Newton")
	plot!(rhist0f[rhist0f.>0], label="Fixed point")

	plot(p1, p2, layout=@layout [a b])
end

# ╔═╡ b9c39280-a19e-4026-9849-5a7768134b10
test_autocatalytic_fp()

# ╔═╡ 276a6f92-d2d7-4cbb-82a1-de440ce44376
md"""
#### Questions

1. Consider choosing $v = \alpha x (1-x)$ so that the equation is exactly satisfied at $x = 1/2$.  How would you do this num
2. Modify the Newton solver for the discretization of the equation $v'' + \lambda \exp(v) = 0$.  What happens as $\lambda$ grows greater than one?  For what size $\lambda$ can you get a solution?  Try incrementally increasing $\lambda$, using the final solution for the previous value of $\lambda$ as the initial guess at the next value.
"""

# ╔═╡ b4684270-efa1-4511-981e-43bbce4c7168
md"""
## Some practical issues

In general, there is no guarantee that a given solution of nonlinear equations will have a solution; and if there is a solution, there is no guarantee of uniqueness. This has a practical implication: many incautious computationalists have been embarrassed to find that they have “solved” a problem that was later proved to have no solution!

When we have reason to believe that a given system of equations has a solution — whether through mathematical argument or physical intuition — we still have the issue of finding a good enough initial estimate that Newton’s method will converge. In coming lectures, we will discuss “globalization” methods that expand the set of initial guesses for which Newton’s method converges; but globalization does not free us from the responsibility of trying for a good guess. Finding a good guess helps ensure that our methods will converge quickly, and to the “correct” solution (particularly when there are multiple possible solutions).

We saw one explicit example of the role of the initial guess in our analysis of the discretized blowup PDE problem. Another example occurs when we use unguarded Newton iteration for optimization. Given a poor choice of initial guess, we are as likely to converge to a saddle point or a local maximum as to a minimum! But we will address this pathology in our discussion of globalization methods.

If we have a nice problem and an adequate initial guess, Newton’s iteration can converge quite quickly. But even then, we still have to think about when we will be satisfied with an approximate solution. A robust solver should check a few possible termination criteria:

- *Iteration count*: It makes sense to terminate (possibly with a diagnostic message) whenever an iteration starts to take more steps than one expects — or perhaps more steps than one can afford. If nothing else, this is necessary to deal with issues like poor initial guesses.

- *Residual check*: We often declare completion when $\|f(x^k)\|$ is sufficiently close to zero. What is “close to zero” depends on the scaling of the problem, so users of black box solvers are well advised to check that any default residual checks make sense for their problems.

- *Update check*: Once Newton starts converging, a good estimate for the error at step $x^k$ is $x^{k+1}-x^k$. A natural test is then to make sure that $\|x^{k+1}-x^k\|/\|x^{k+1}\| < \tau$ for some tolerance $\tau$. Of course, this is really an estimate of the relative error at step $k$, but we will report the (presumably better) answer $x^{k+1}$ -- like anyone else who can manage it, numerical analysts like to have their cake and eat it, too.

A common problem with many solvers is to make the termination criteria too lax, so that a bad solution is accepted; or too conservative, so that good solutions are never accepted.

One common mistake in coding Newton’s method is to goof in computing the Jacobian matrix. This error is not only very common, but also very often overlooked. After all, a good approximation to the Jacobian often still produces a convergent iteration; and when iterations diverge, it is hard to distinguish between problems due to a bad Jacobian and problems due to a bad initial guess. However, there is a simple clue to watch for that can help alert the user to a bad Jacobian calculation. In most cases, Newton converges quadratically, while “almost Newton” iterations converge linearly. If you think you have coded Newton’s method and a plot of the residuals shows linear behavior, look for issues with your Jacobian code!
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[compat]
Plots = "~1.27.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "a4e743db5584ae12092350b26bdabc7966ad8fa2"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

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
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

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
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "9f836fb62492f4b0f0d3b06f55983f2704ed0883"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a6c850d77ad5118ad3be4bd188919ce97fffac47"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.0+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

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
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

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
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "4f00cc36fede3c04b8acf9b2e2763decfdcecfa6"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.13"

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
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

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
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

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
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

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

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

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
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "5f6e1309595e95db24342e56cd4dabd2159e0b79"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

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
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

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
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "4f6ec5d99a28e1a749559ef7dd518663c5eca3d5"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

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
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

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
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─c52a00f8-ae1e-11ec-39af-5504c28148bf
# ╠═0e0fc087-fdb7-42a8-8e55-1883faf4b831
# ╠═d24fb1e0-2606-4f39-b8c2-35a6c6c6319e
# ╟─81180297-5e55-4ee0-910f-72be07a18b06
# ╟─0a85f9e5-d964-4c20-993a-2a902c86cd70
# ╠═50492436-7209-4bfc-92f3-557ba0db4ad9
# ╠═090659d5-f155-4b67-af45-c823e782ba12
# ╟─7985ec7e-29b0-4cbb-a7b5-b0d650badfd1
# ╟─23e8fd18-41be-46e4-83cc-778cc5e29849
# ╟─0058da6b-fea2-495f-905a-44f13398a98a
# ╠═662f3ec9-5cf0-4ea5-a6ad-fd9ce8fa8595
# ╠═81e9aaf3-bfa5-4247-a518-a08e3f7d1c34
# ╠═371c69b8-8d71-4f5d-b131-394e16f4988c
# ╟─60857857-aeae-4255-af74-64c535b7d66d
# ╟─99ba3897-4b14-4ef3-894e-48d6ad95a329
# ╟─00f65731-867d-421e-b38d-bbe8a6b0ffcd
# ╠═1a98df17-3fd6-4ccf-8375-60eab7a6556d
# ╟─e98e962c-df4e-439f-bfec-611e298a6ea1
# ╠═ee0d40b8-3114-4093-8f57-49cb7ed8b0fc
# ╟─cfdbc815-00d9-4fff-a921-217c845fbc82
# ╠═fb4b9029-b474-4f94-9863-e4649548c076
# ╠═e9dfc16b-7c22-4154-a819-011872ad2184
# ╠═58932838-0a88-4866-b1a0-bb2e83eb1737
# ╠═8e4bc440-e98a-46c2-ba80-7b2016c61587
# ╟─3b2c56f0-5725-45b6-bbdc-71e4aaa15e03
# ╠═84cdc15b-9cdd-4462-a306-e64352df4518
# ╠═c584e561-0b45-4274-affe-746fcccaa0e2
# ╠═163e3d56-ad19-4256-9830-3eae026cbdc0
# ╠═b9c39280-a19e-4026-9849-5a7768134b10
# ╟─276a6f92-d2d7-4cbb-82a1-de440ce44376
# ╟─b4684270-efa1-4511-981e-43bbce4c7168
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
