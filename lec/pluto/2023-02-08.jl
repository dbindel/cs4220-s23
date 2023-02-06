### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ feb08f83-c795-45c9-bc6e-d71175f271de
using LinearAlgebra

# ╔═╡ 3b0eb41c-945e-11ec-0e1f-7902c17dbb78
md"""
# Notebook for 2023-02-08

In this lecture, we discussed forward and backward substitution for triangular matrices, and the basic Gaussian elimination algorithm without pivoting.  We copy in the algorithms here and illustrate them, as well as illustrating what we would really want to do in Julia using the `LinearAlgebra` package rather than rolling our own.
"""

# ╔═╡ 89835dbc-b472-4f1e-bce0-2cd1a9e473c6
md"""
## Triangular solves

LU factorization gives us a *unit* lower triangular matrix and an upper triangular matrix as factors, so let's start by figuring out how to solve systems with them.
"""

# ╔═╡ 9953f3fb-67a2-4af3-818c-2c3b53a5276b
function forward_subst_unit(L, d)
	y = copy(d)
	n = length(d)
	for j = 1:n
		y[j] -= L[j,1:j-1]'*y[1:j-1]
	end
	y
end

# ╔═╡ 2884fee8-c3e7-4fcb-824c-00365b6db8b5
function test_forward_subst()
	
	# Set up test problem: L is unit lower triangular (diagonals are all 1)
	L = [1.0 0.0 0.0; 
	     2.0 1.0 0.0; 
	     3.0 2.0 1.0]
    d = [1.0; 1.0; 3.0]
	
	y1 = forward_subst_unit(L, d)  # Solve with our routine
	y2 = L\d                       # Solve with the Julia built-in
	y1, y2                         # Return the results for side-by-side viewing
end

# ╔═╡ 9bb5efd4-dff0-4f17-9531-e33446a80ecb
test_forward_subst()

# ╔═╡ d1b30fd4-e71b-407f-9110-c5523f5d78ab
function backward_subst(U, d)
	x = copy(d)
	n = length(d)
	for j = n:-1:1
		x[j] = (d[j] - U[j,j+1:n]'*x[j+1:n])/U[j,j]
	end
	x
end

# ╔═╡ e889fed1-cb11-45a6-8c8b-7e093cbc7d01
function test_backward_subst()
	# Set up test problem: U is upper triangular
	U = [1.0  4.0  7.0; 
	     0.0 -3.0 -6.0; 
	     0.0  0.0  1.0]
    d = [1.0; 1.0; 3.0]
	
	y1 = backward_subst(U, d)  # Solve with our routine
	y2 = U\d                   # Solve with the Julia built-in
	y1, y2                     # Return the results for side-by-side viewing
end

# ╔═╡ a12560eb-27a5-44e6-8fc9-d5b3bd50d60c
test_backward_subst()

# ╔═╡ d1faff77-a489-412b-bdf7-0257660f4ee1
md"""
## Gaussian elimination

Let's write a version of the LU factorization routine that overwrites $A$ with the factors.
"""

# ╔═╡ 74fa2a63-4290-4551-92f3-3d90fdaf8a9f
function my_lu!(A)
	n = size(A)[1]
	for j = 1:n
		A[j+1:n,j] /= A[j,j]                      # Multipliers (column of L)
		A[j+1:n,j+1:n] -= A[j+1:n,j]*A[j,j+1:n]'  # Schur complement update
	end
	A
end

# ╔═╡ c6ef124c-5b14-4ecd-b9b7-bb25ef7af06d
md"""
It's convenient to also have a version that does not overwrite the underlying matrix.
"""

# ╔═╡ 6dbe56eb-9f90-43c3-bcf2-c0f2788e8cae
function my_lu(A)
	LU = my_lu!(copy(A))
	L = UnitLowerTriangular(LU)
	U = UpperTriangular(LU)
	L, U
end

# ╔═╡ 3ba244c6-7844-40d1-9ace-93c7a293dbf7
function test_my_lu()
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 10.0]
	L, U = my_lu(A)
	norm(A-L*U)/norm(A)  # Return relative residual error (Frobenius norm)
end

# ╔═╡ ab237b15-e1ff-4ecd-9fb9-76da2987fa64
test_my_lu()

# ╔═╡ 61844e20-dacf-4da2-b7e7-73f2d55333e0
md"""
If we want to solve linear systems, we can do forward and backward substitution, either using the triangular views or directly on the packed form of the factorization.
"""

# ╔═╡ f4a81170-6fee-4817-b904-30db7a40aa45
function test_lu_solve1()

	# Manufacture a linear system
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 10.0]
	x = rand(3)
	b = A*x

	# Factor and solve
	my_lu!(A)
	x2 = backward_subst(A, forward_subst_unit(A, b))

	# Return a relative error
	norm(x2-x)/norm(x)
end

# ╔═╡ 9f03e09d-ca26-451e-ad30-fe82d360d64a
test_lu_solve1()

# ╔═╡ 944e1356-e591-4540-b62f-27ffc051646b
function test_lu_solve2()

	# Manufacture a linear system
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 10.0]
	x = rand(3)
	b = A*x

	# Factor and solve (use built-in triangular solvers)
	L, U = my_lu(A)
	x2 = U\(L\b)

	# Return a relative error
	norm(x2-x)/norm(x)
end

# ╔═╡ 1c667ce2-7174-4e05-85ad-6def8aad9862
test_lu_solve2()

# ╔═╡ 80047094-c436-403a-98e2-e12b097f374a
md"""
Finally, let's look at how to do the whole thing end-to-end with the built-in LU routines in Julia.  The factorization we compute in this case includes pivoting for stability, which is why the $L$ and $U$ factors are different from those before.
"""

# ╔═╡ 424c793f-fa5c-4a12-af24-0f83a728e7ef
begin
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 10.0]
	x = rand(3)
	b = A*x

	F = lu(A)
end

# ╔═╡ 7b31868c-f2b4-4280-b702-68ba6b225af3
md"""
The result with pivoting should look like $PA = LU$; note that Julia stores the permutation as a reordered index vector rather than as a matrix.
"""

# ╔═╡ 6ac0e743-30a2-4bb9-89ca-7199571c212a
A[F.p,:] - F.L*F.U

# ╔═╡ b364d8f4-fff0-4ff5-939a-3cc63cc63f4a
md"""
We can solve the linear system either by using the factorization object or by using the pieces of the factorization (as we did before).  The former approach is usually simpler.
"""

# ╔═╡ 75bb8810-4b12-4194-834d-db3796d9cbf8
x-F\b

# ╔═╡ cde5da79-f5cf-49be-b5da-45f4285279d2
x-F.U\(F.L\b[F.p])

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
# ╟─3b0eb41c-945e-11ec-0e1f-7902c17dbb78
# ╠═feb08f83-c795-45c9-bc6e-d71175f271de
# ╟─89835dbc-b472-4f1e-bce0-2cd1a9e473c6
# ╠═9953f3fb-67a2-4af3-818c-2c3b53a5276b
# ╠═2884fee8-c3e7-4fcb-824c-00365b6db8b5
# ╠═9bb5efd4-dff0-4f17-9531-e33446a80ecb
# ╠═d1b30fd4-e71b-407f-9110-c5523f5d78ab
# ╠═e889fed1-cb11-45a6-8c8b-7e093cbc7d01
# ╠═a12560eb-27a5-44e6-8fc9-d5b3bd50d60c
# ╟─d1faff77-a489-412b-bdf7-0257660f4ee1
# ╠═74fa2a63-4290-4551-92f3-3d90fdaf8a9f
# ╟─c6ef124c-5b14-4ecd-b9b7-bb25ef7af06d
# ╠═6dbe56eb-9f90-43c3-bcf2-c0f2788e8cae
# ╠═3ba244c6-7844-40d1-9ace-93c7a293dbf7
# ╠═ab237b15-e1ff-4ecd-9fb9-76da2987fa64
# ╟─61844e20-dacf-4da2-b7e7-73f2d55333e0
# ╠═f4a81170-6fee-4817-b904-30db7a40aa45
# ╠═9f03e09d-ca26-451e-ad30-fe82d360d64a
# ╠═944e1356-e591-4540-b62f-27ffc051646b
# ╠═1c667ce2-7174-4e05-85ad-6def8aad9862
# ╟─80047094-c436-403a-98e2-e12b097f374a
# ╠═424c793f-fa5c-4a12-af24-0f83a728e7ef
# ╟─7b31868c-f2b4-4280-b702-68ba6b225af3
# ╠═6ac0e743-30a2-4bb9-89ca-7199571c212a
# ╟─b364d8f4-fff0-4ff5-939a-3cc63cc63f4a
# ╠═75bb8810-4b12-4194-834d-db3796d9cbf8
# ╠═cde5da79-f5cf-49be-b5da-45f4285279d2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
