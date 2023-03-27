### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 7e3384bf-5aeb-47b1-8a9e-d1b32aa0417f
using LinearAlgebra

# ╔═╡ d0f201bd-f761-497c-a986-cdce3ede2c8f
md"""
# Notebook for 2023-02-13
"""

# ╔═╡ 21dff7ce-9d70-11ec-04ef-2fc574438500
md"""
## Pivoting

Row pivoting is necessary for Gaussian elimination in exact arithmetic because of the possibility that a leading submatrix will be singular.  For example

$$\begin{bmatrix} 0 & 1 \\ 1 & 1 \end{bmatrix}$$

does not admit an unpivoted LU decomposition.  If we perturb the (1,1) entry, we have a nearby matrix that *can* be factored without pivoting:

$$\begin{bmatrix} \delta & 1 \\ 1 & 1 \end{bmatrix} =
  \begin{bmatrix} 1 & 0 \\ \delta^{-1} & 1 \end{bmatrix}
  \begin{bmatrix} \delta & 1 \\ 0 & 1-\delta^{-1} \end{bmatrix}.$$

But this factorization has terrible backward error, which we can compare to the (approximate) bound derived in class: $|\hat{A}-A| \leq n \epsilon_{\mathrm{mach}} |L| |U|$ elementwise.  Note that what we refer to as machine epsilon in class is sometimes called the unit roundoff, which is half the distance between 1.0 and the next largest floating point number -- the Julia expression `eps(Float64)` is the distance between 1.0 and the next largest floating point number, which is twice as big.
"""

# ╔═╡ 941c0ee5-2534-4243-9a8a-b490ffe7a158
begin
	δ = 1.0e-16
	Abad = [δ 1.0; 1.0 1.0]
	Lbad = [1.0 0.0; 1.0/δ 1.0]
	Ubad = [δ 1.0; 0.0 1.0-1.0/δ]
	Abad-Lbad*Ubad, eps(Float64)*abs.(Lbad)*abs.(Ubad)
end

# ╔═╡ 6bdcc960-9113-408f-8e56-cd3940be211f
md"""
The usual strategy of *partial pivoting* chooses to eliminate variable $j$ using whichever row has the largest element in column $i$ of the Schur complement.  This guarantees that the entries of $L$ are all bounded by 1 in magnitude.
"""

# ╔═╡ 14c15261-c5e4-4530-80e0-41671826e0c7
function my_pivoted_lu(A)

	n = size(A)[1]
	A = copy(A)         # Make a local copy to overwrite
	piv = zeros(Int, n) # Space for the pivot vector
	piv[1:n] = 1:n

	for j = 1:n-1

		# Find ipiv >= j to maximize |A(i,j)|
		ipiv = (j-1)+findmax(abs.(A[j:n,j]))[2]

		# Swap row ipiv and row j and record the pivot row
		A[ipiv,:], A[j,:] = A[j,:], A[ipiv,:]
		piv[ipiv], piv[j] = piv[j], piv[ipiv]

		# Compute multipliers and update trailing submatrix
		A[j+1:n,j] /= A[j,j]
		A[j+1:n,j+1:n] -= A[j+1:n,j]*A[j,j+1:n]'
		
	end
	
	UnitLowerTriangular(A), UpperTriangular(A), piv
end

# ╔═╡ e2d956d2-91b6-492f-89fa-4bba43c49d3d
begin
	A = rand(10,10)
	b = rand(10)
	L, U, p = my_pivoted_lu(A)
	norm(A[p,:]-L*U)/norm(A), maximum(abs.(L))
end

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
# ╟─d0f201bd-f761-497c-a986-cdce3ede2c8f
# ╠═7e3384bf-5aeb-47b1-8a9e-d1b32aa0417f
# ╟─21dff7ce-9d70-11ec-04ef-2fc574438500
# ╠═941c0ee5-2534-4243-9a8a-b490ffe7a158
# ╟─6bdcc960-9113-408f-8e56-cd3940be211f
# ╠═14c15261-c5e4-4530-80e0-41671826e0c7
# ╠═e2d956d2-91b6-492f-89fa-4bba43c49d3d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
