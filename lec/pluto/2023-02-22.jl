### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 06297bfa-ddde-4bca-91e6-788ae7ad3abb
using LinearAlgebra

# ╔═╡ 462b5032-94a7-11ec-0b78-733a9f46100d
md"""
# Notes for 2023-02-22

## Gram-Schmidt

The *Gram-Schmidt* procedure is usually the first method people
learn to convert some existing basis (columns of $A$) into an
orthonormal basis (columns of $Q$).  For each column of $A$, the procedure
subtracts off any components in the direction of the previous columns,
and then scales the remainder to be unit length. In Julia, Gram-Schmidt looks
something like this:
"""

# ╔═╡ 9308e0c8-2efb-45fc-8a38-d8f1fba1e762
function orth_cgs0(A)
	m,n = size(A)
	Q = zeros(m,n)
	for j = 1:n
		v = A[:,j]                        # Take the jth original basis vector
		v = v-Q[:,1:j-1]*(Q[:,1:j-1]'*v)  # Orthogonalize vs q_1, ... q_j-1
		v = v/norm(v)                     # Normalize what remains
		Q[:,j] = v                        # Add result to Q basis
	end
	Q
end

# ╔═╡ b41b384f-76c4-4d76-906f-2bde8907139b
function test_orth_cgs0()
	A = rand(10,5)
	Q = orth_cgs0(A)

	# Check orthonormality of Q columns, and that Q gives the right projector
	norm(Q'*Q-I), norm(A-Q*(Q'*A))/norm(A)
end

# ╔═╡ f5eef595-85f8-44f7-a62f-fe43ba8549b2
test_orth_cgs0()

# ╔═╡ 60b651ef-552c-471b-9a42-ebc09c4f9a43
md"""
Where does $R$ appear in this algorithm?  As with the lower triangle in the LU factorization, the $R$ factor in QR appears implicitly in the orthogonalization and normalization steps.
"""

# ╔═╡ 865bfa76-9cdb-4f1e-b29f-886cbed2ed5f
function orth_cgs(A)
	m,n = size(A)
	Q = zeros(m,n)
	R = zeros(n,n)
	for j = 1:n
		v = A[:,j]                        # Take the jth original basis vector
		R[1:j-1,j] = Q[:,1:j-1]'*v        # Project onto q_1, ..., q_j-1
		v = v-Q[:,1:j-1]*R[1:j-1,j]       # Orthogonalize vs q_1, ... q_j-1
		R[j,j] = norm(v)                  # Compute normalization constant
		v = v/R[j,j]                      # Normalize what remains
		Q[:,j] = v                        # Add result to Q basis
	end
	Q, R
end

# ╔═╡ d6dd0bce-fad6-45e0-a947-cc74df7844c7
function test_orth_cgs()
	A = rand(10,5)
	Q, R = orth_cgs(A)

	# Check orthonormality of Q columns, 
	#       upper triangularity of R,
	#       and that A = QR
	norm(Q'*Q-I), norm(tril(R,-1)), norm(A-Q*R)/norm(A)
end

# ╔═╡ bbf5420d-180d-439d-8bd7-e89c87b793b7
test_orth_cgs()

# ╔═╡ 4502b824-51dd-4839-b04d-df1f39e8d5d4
md"""
Testing with random matrices is a particularly gentle way to test our code, and doesn't reveal a lurking numerical instability in the CGS algorithm.  The problem is that if $A$ is even somewhat ill conditioned (i.e. the basis vectors in $A$ are nearly co-linear) then the $Q$ computed by the classical Gram-Schmidt algorithm may stop being nearly orthogonal, even if the relationship $A = QR$ is well maintained.
"""

# ╔═╡ 16544c2d-3948-4218-bd24-af313176e030
function test_orth_cgs_poor()
	A = rand(10,5)*Diagonal([1, 1e-2, 1e-4, 1e-6, 1e-8])*rand(5,5)
	Q, R = orth_cgs(A)

	# Check orthonormality of Q columns, 
	#       upper triangularity of R,
	#       and that A = QR
	norm(Q'*Q-I), norm(tril(R,-1)), norm(A-Q*R)/norm(A)
end

# ╔═╡ bb7ae656-8d2b-4a35-9c56-462ccafe2718
test_orth_cgs_poor()

# ╔═╡ 69cb0e71-00c5-4c45-bab6-ba71e61efe99
md"""
This is somewhat ameliorated by the *modified* Gram-Schmidt algorithm, which computes the multipliers in $R$ in a slightly different way: subtracting off the projections of the residual onto each previously computed column of $k$ immediately rather than computing all the projections from the original vector.  This is a little less prone to cancellation.  We can see that numerical orthogonality with MGS is much better than with CGS, though there is still some loss.
"""

# ╔═╡ bf55ce8b-4e0a-4fb2-90af-07a98cd1e032
function orth_mgs(A)
	m,n = size(A)
	Q = zeros(m,n)
	R = zeros(n,n)
	for j = 1:n
		v = A[:,j]                        # Take the jth original basis vector
		for k = 1:j-1
			R[k,j] = Q[:,k]'*v            # Project what remains onto q_k
			v -= Q[:,k]*R[k,j]            # Subtract off the q_k component
		end
		R[j,j] = norm(v)                  # Compute normalization constant
		v = v/R[j,j]                      # Normalize what remains
		Q[:,j] = v                        # Add result to Q basis
	end
	Q, R
end

# ╔═╡ 6df81b32-9401-4f39-b1c3-5425067fb0b4
function test_orth_mgs_poor()
	A = rand(10,5)*Diagonal([1, 1e-2, 1e-4, 1e-6, 1e-8])*rand(5,5)
	Q, R = orth_mgs(A)

	# Check orthonormality of Q columns, 
	#       upper triangularity of R,
	#       and that A = QR
	norm(Q'*Q-I), norm(tril(R,-1)), norm(A-Q*R)/norm(A)
end

# ╔═╡ b421b8fd-bdc3-4021-95ba-378606022a05
test_orth_mgs_poor()

# ╔═╡ bfa6d77b-43d9-4383-b1f7-2e167626de4c
md"""
## Householder transformations

A *Householder transformation* is a simple orthogonal transformation associated with reflection across a plane with some unit normal $v$:

$$H = I-2vv^T$$

We typically choose the transformation so that we can map a target vector (say $x$) to something parallel to the first coordinate axis (so $\pm \|x\| e_1$).  In some cases, it is convenient to use a *non-unit* normal $w$ and introduce an associated scaling constant:

$$H = I - \tau ww^T$$

This lets us normalize $w$ in other ways (e.g. by putting a 1 in the first component).
"""

# ╔═╡ 16c408ad-68f6-4d37-981b-cdf69da71cd5
function householder(x)
	normx = norm(x)
	s = -sign(x[1])
	u1 = x[1]-s*normx
	w = x/u1
	w[1] = 1.0
	τ = -s*u1/normx
	τ, w
end

# ╔═╡ 519d7d9f-740e-43ab-ad1f-00b29a79e8e9
function test_householder()
	x = rand(10)
	τ, w = householder(x)

	# Check orthogonality and the mapping of x
	norm((I-τ*w*w')^2-I), norm((x-τ*w*(w'*x))[2:end])/norm(x)
end

# ╔═╡ af2602c2-533b-423e-9952-0336f158c6d3
test_householder()

# ╔═╡ b19708a0-fa63-4a1c-bf76-92bb36c09353
md"""
Now we can think about QR in the same way we think about LU.  The only difference is that we apply a sequence of Householder transformations (rather than Gauss transformations) to introduce the subdiagonal zeros.  As with the LU factorization, we can also reuse the storage of A by recognizing that the number of nontrivial parameters in the vector $w$ at each step is the same as the number of zeros produced by that transformation.  This gives us the following.
"""

# ╔═╡ e72043ab-05f5-4684-9227-f55e41f03335
function hqr!(A)
	m,n = size(A)
	τ = zeros(n)

	for j = 1:n

		# Find H = I-τ*w*w' to zero out A[j+1:end,j]
		normx = norm(A[j:end,j])
		s     = -sign(A[j,j])
		u1    = A[j,j] - s*normx
		w     = A[j:end,j]/u1
		w[1]  = 1.0
		A[j+1:end,j] = w[2:end]   # Save trailing part of w
		A[j,j] = s*normx          # Diagonal element of R
		τ[j] = -s*u1/normx        # Save scaling factor

		# Update trailing submatrix by multipling by H
		A[j:end,j+1:end] -= τ[j]*w*(w'*A[j:end,j+1:end])

	end

	A, τ
end

# ╔═╡ 2d6603d0-e83a-4e11-9667-a6ff6aeb9145
md"""
If we need $Q$ or $Q^T$ explicitly, we can always form it from the compressed representation.  We can also multiply by $Q$ and $Q^T$ implicitly.
"""

# ╔═╡ 2f43b935-6c19-4412-be38-90e0992abc33
function applyQ!(QR, τ, X)
	m, n = size(QR)
	for j = n:-1:1
		w = [1.0; QR[j+1:end,j]]
		X[j:end,:] -= τ[j]*w*(w'*X[j:end,:])
	end
	X
end

# ╔═╡ eeb2bd27-f5ce-4756-b19f-43608827e88b
function applyQT!(QR, τ, X)
	m, n = size(QR)
	for j = 1:n
		w = [1.0; QR[j+1:end,j]]
		X[j:end,:] -= τ[j]*w*(w'*X[j:end,:])
	end
	X
end

# ╔═╡ 0b07e58b-f9ef-4e9b-96c8-84286b1770ac
applyQ(QR, τ, X) = applyQ!(QR, τ, copy(X))

# ╔═╡ e4a32e05-a9f6-4a32-91a4-302e63dfaa4d
applyQT(QR, τ, X) = applyQ(QR, τ, copy(X))

# ╔═╡ d7eb1227-24f8-4a05-9282-67dba037b214
md"""
To compute a dense representation of $Q$, we can always apply $Q$ to the columns of the identity.
"""

# ╔═╡ a67fb273-3ed8-480a-831e-adf1e29dca19
formQ(QR, τ) = applyQ!(QR, τ, Matrix(1.0I, size(QR)[1], size(QR)[1]))

# ╔═╡ 1cc34294-82e8-45c2-b740-f638f3f049f4
md"""
This gives us all the ingredients to form a dense QR decomposition of $A$.
"""

# ╔═╡ 06906287-dd90-447a-b034-6c000759b287
function hqr(A)
	QR, τ = hqr!(copy(A))
	formQ(QR, τ), triu(QR)
end

# ╔═╡ 4509f664-e2ab-4226-8aea-55fd9d64d0b4
function test_hqr()
	A = rand(10,5)
	Q, R = hqr(A)
	norm(Q'*Q-I), norm(A-Q*R)/norm(A)
end

# ╔═╡ 91cbc2ef-e53c-427b-8dfb-e4eea3f39703
test_hqr()

# ╔═╡ eca0ce75-924c-4f35-a34f-e7281e204efc
md"""
However, we don't need the dense $Q$ and $R$ factors to solve a least squares problem; we just need the ability to multiply by $Q^T$ and solve with $R$.
"""

# ╔═╡ a8c18535-be12-451a-9be1-a1a905f38bfa
function hqr_solve_ls(QR, τ, b)
	m, n = size(QR)
	UpperTriangular(QR[1:n,1:n])\(applyQT!(QR, τ, copy(b))[1:n])
end

# ╔═╡ b70e82c7-8212-4ec3-982b-40d8f7bde1f3
function test_hqr_solve_ls()
	A = rand(10,5)
	b = rand(10)

	QR, τ = hqr!(copy(A))

	xref = A\b
	x = hqr_solve_ls(QR, τ, b)
	norm(x-xref)/norm(xref)
end

# ╔═╡ 513c2328-a3aa-4c0b-88ab-aff432aeb76f
test_hqr_solve_ls()

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
# ╟─462b5032-94a7-11ec-0b78-733a9f46100d
# ╠═06297bfa-ddde-4bca-91e6-788ae7ad3abb
# ╠═9308e0c8-2efb-45fc-8a38-d8f1fba1e762
# ╠═b41b384f-76c4-4d76-906f-2bde8907139b
# ╠═f5eef595-85f8-44f7-a62f-fe43ba8549b2
# ╟─60b651ef-552c-471b-9a42-ebc09c4f9a43
# ╠═865bfa76-9cdb-4f1e-b29f-886cbed2ed5f
# ╠═d6dd0bce-fad6-45e0-a947-cc74df7844c7
# ╠═bbf5420d-180d-439d-8bd7-e89c87b793b7
# ╟─4502b824-51dd-4839-b04d-df1f39e8d5d4
# ╠═16544c2d-3948-4218-bd24-af313176e030
# ╠═bb7ae656-8d2b-4a35-9c56-462ccafe2718
# ╟─69cb0e71-00c5-4c45-bab6-ba71e61efe99
# ╠═bf55ce8b-4e0a-4fb2-90af-07a98cd1e032
# ╠═6df81b32-9401-4f39-b1c3-5425067fb0b4
# ╠═b421b8fd-bdc3-4021-95ba-378606022a05
# ╟─bfa6d77b-43d9-4383-b1f7-2e167626de4c
# ╠═16c408ad-68f6-4d37-981b-cdf69da71cd5
# ╠═519d7d9f-740e-43ab-ad1f-00b29a79e8e9
# ╠═af2602c2-533b-423e-9952-0336f158c6d3
# ╟─b19708a0-fa63-4a1c-bf76-92bb36c09353
# ╠═e72043ab-05f5-4684-9227-f55e41f03335
# ╟─2d6603d0-e83a-4e11-9667-a6ff6aeb9145
# ╠═2f43b935-6c19-4412-be38-90e0992abc33
# ╠═eeb2bd27-f5ce-4756-b19f-43608827e88b
# ╠═0b07e58b-f9ef-4e9b-96c8-84286b1770ac
# ╠═e4a32e05-a9f6-4a32-91a4-302e63dfaa4d
# ╟─d7eb1227-24f8-4a05-9282-67dba037b214
# ╠═a67fb273-3ed8-480a-831e-adf1e29dca19
# ╟─1cc34294-82e8-45c2-b740-f638f3f049f4
# ╠═06906287-dd90-447a-b034-6c000759b287
# ╠═4509f664-e2ab-4226-8aea-55fd9d64d0b4
# ╠═91cbc2ef-e53c-427b-8dfb-e4eea3f39703
# ╟─eca0ce75-924c-4f35-a34f-e7281e204efc
# ╠═a8c18535-be12-451a-9be1-a1a905f38bfa
# ╠═b70e82c7-8212-4ec3-982b-40d8f7bde1f3
# ╠═513c2328-a3aa-4c0b-88ab-aff432aeb76f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
