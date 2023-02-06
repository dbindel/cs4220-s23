### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 8ecf77f0-9aa7-4815-a02f-ac3bf260e63e
using LinearAlgebra

# ╔═╡ cda3a5c0-9464-11ec-3e04-c391760d8527
md"""
# Notebook for 2023-02-10

## Block Gaussian elimination

The main topic for this lecture was thinking about Gaussian elimination in a blockwise fashion.  In particular, we considered solving block 2-by-2 linear systems with matrices of the form

$$A = \begin{bmatrix} B & V \\ W^T & C \end{bmatrix}$$

A particular instance of this is solving the linear system

$$\begin{bmatrix} B & V \\ W^T & -I \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} d \\ 0 \end{bmatrix}$$

In terms of two block equations, this is

$$\begin{aligned}
Bx + Vy &= d \\ 
W^T x - y &= 0
\end{aligned}$$

and substituting $y = W^T x$ (from the latter equation) into the former equation gives

$$(B + VW^T)x = d$$

Going the other way, we can eliminate the $x$ variables to get an equation just in $y$:

$$(-I-W^T B^{-1} V) y = -W^T B^{-1} d$$

and then substitute into $Bx = d-Vy$ to solve the system.

Let's go ahead and illustrate how to use this to solve $(B + VW^T) x = d$ given a solver for systems with $B$.
"""

# ╔═╡ 35b6c6e0-da7c-4bc9-b921-8912c152d8f5
# Solves the system (B+VW') x = d given a solver solveB(g) = B\g
function solve_bordered(solveB, V, W, d)
	invBV = solveB(V)
	invBd = solveB(d)
	S = I + W'*invBV
	y = S\(W'*invBd)
	invBd - invBV*y
end

# ╔═╡ 6c46876e-b8ea-4c04-82df-519ddfce4921
function test_solve_bordered()
	B = rand(10,10)
	V = rand(10,2)
	W = rand(10,2)
	x = rand(10)
	d = B*x+V*(W'*x)
	solveB(y) = B\y
	err = x - solve_bordered(solveB, V, W, d)
	norm(err)/norm(x)
end

# ╔═╡ 721050c0-3233-4f2f-b1db-f70190167c77
test_solve_bordered()

# ╔═╡ cb6998db-4829-4482-bd2a-e3296233bbe7
md"""
## Iterative refinement

The trick of forming a bordered system is interesting in part because it allows us to take advantage of a fast solver for one system (from an existing factorization, for example) in order to solve a slightly modified system.  But there's a cost!  Even if the modified system is well-conditioned, poor conditioning in the reference problem (the $B$ matrix) can lead to quite bad results.
"""

# ╔═╡ 62b7efd3-7bd2-4312-ae81-30a1e4cd8b2d
function test_sensitive_bordered()
	s = ones(10)
	s[10] = 1e-12
	B = rand(10,10)*Diagonal(s)*rand(10,10)
	V = rand(10,2)
	W = rand(10,2)
	x = rand(10)
	d = B*x+V*(W'*x)
	solveB(y) = B\y
	err = x - solve_bordered(solveB, V, W, d)
	norm(err)/norm(x)
end

# ╔═╡ 622acdd7-9d50-4b50-8ec0-5366ec3a8d93
test_sensitive_bordered()

# ╔═╡ eb921400-8578-446d-99a1-21dddb3f440d
md"""
A technique in the notes (but not discussed in lecture) can be used to partly fix up this bad behavior.  When we have a somewhat sloppy solver like the one here, but we're able to compute a good residual (and the problem we're solving is not terribly ill conditioned) sometimes a few steps of *iterative refinement* can make a big difference.
"""

# ╔═╡ 849af83a-de0c-4187-a583-87c8f82bba41
function test_itref()
	s = ones(10)
	s[10] = 1e-12
	B = rand(10,10)*Diagonal(s)*rand(10,10)
	V = rand(10,2)
	W = rand(10,2)
	x = rand(10)
	d = B*x+V*(W'*x)
	solveB(y) = B\y

	relerrs = zeros(5)
	xx = solve_bordered(solveB, V, W, d)
	relerrs[1] = norm(x-xx)/norm(x)
	for j = 2:5
		xx += solve_bordered(solveB, V, W, d-B*xx-V*(W'*xx))
		relerrs[j] = norm(x-xx)/norm(x)
	end
	relerrs
end

# ╔═╡ c2a8abd2-9b60-4aaf-9769-c1f05b36ddb8
test_itref()

# ╔═╡ fc38cf8e-2b7f-4d73-ba95-1a8553581bb7
md"""
## Sensitivity

We also spent a little time in this lecture talking about the sensitivity of linear systems to perturbations.  In particular, we differentiated the equation

$$Ax = b$$

which, using variational notation, gave us

$$(\delta A) x + A (\delta x) = \delta b$$

In addition to being useful for error analysis, this formula is actually something you can use computationally, as we illustrate below.
"""

# ╔═╡ d6b1f303-a3bc-4ee8-81a1-75b182d3646a
function test_perturbed()

	# Set up a test problem and directions to move A and b
	A  = rand(10,10)
	δA = rand(10,10)
	b  = rand(10)
	δb = rand(10)

	# Compute solutions to a reference and perturbed system
	h = 1e-8
	x = A\b
	x̂ = (A+h*δA)\(b+h*δb)

	# Compute the directional deriv δx and a finite difference estimate
	δx_est = (x̂-x)/h
	δx = A\(δb-δA*x)

	# Compare the analytic derivative to the finite difference estimate
	norm(δx-δx_est)/norm(δx)
end

# ╔═╡ 77d8c552-c020-4cf8-b48e-565984e85747
test_perturbed()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "ac1187e548c6ab173ac57d4e72da1620216bce54"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"
"""

# ╔═╡ Cell order:
# ╟─cda3a5c0-9464-11ec-3e04-c391760d8527
# ╠═8ecf77f0-9aa7-4815-a02f-ac3bf260e63e
# ╠═35b6c6e0-da7c-4bc9-b921-8912c152d8f5
# ╠═6c46876e-b8ea-4c04-82df-519ddfce4921
# ╠═721050c0-3233-4f2f-b1db-f70190167c77
# ╟─cb6998db-4829-4482-bd2a-e3296233bbe7
# ╠═62b7efd3-7bd2-4312-ae81-30a1e4cd8b2d
# ╠═622acdd7-9d50-4b50-8ec0-5366ec3a8d93
# ╟─eb921400-8578-446d-99a1-21dddb3f440d
# ╠═849af83a-de0c-4187-a583-87c8f82bba41
# ╠═c2a8abd2-9b60-4aaf-9769-c1f05b36ddb8
# ╟─fc38cf8e-2b7f-4d73-ba95-1a8553581bb7
# ╠═d6b1f303-a3bc-4ee8-81a1-75b182d3646a
# ╠═77d8c552-c020-4cf8-b48e-565984e85747
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
