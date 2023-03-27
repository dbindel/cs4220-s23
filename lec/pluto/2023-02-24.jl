### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ ae5453a7-6d80-4fa3-9475-2320cec847cc
using LinearAlgebra

# ╔═╡ 0064d806-9d7c-11ec-0a5b-052a00b0c0bb
md"""
# Notebook for 2023-02-24

The main topic of this lecture is regularization for ill-posed systems.  We will illustrate this with an *exactly* singular problem.
"""

# ╔═╡ d8cfd77b-05fe-4b6f-a78b-8304901f5977
begin

	# Exact and noisy factors
	A = randn(100,2) * randn(2,4)
	E1 = 1e-12 * rand(10,4)
	E2 = 1e-8 * randn(90,4)
	E = [E1; E2]
	Â = A+E
	Â1 = Â[1:10,:]

	# Reference coefficients
	c = [1.0; 2.0; 3.0; 4.0]

	# Exact and noisy rhs
	b = A*c
	η = 1e-3 * randn(100)
	b̂ = b + η
	b̂1 = b̂[1:10]

end

# ╔═╡ ba8c2659-893c-47ba-b194-0f8e9b9992ee
md"""
If we are given the noisy matrix $\hat{A}$ and the noisy right hand side $\hat{b}$, we will basically always get something terrible, because we are fitting the noise.
In particular, notice the giant coefficients if we fit with just the first part of the data
"""

# ╔═╡ 928f6a09-f65c-4dbf-bf5c-768e0bcc5c80
ĉ = Â1\b̂1

# ╔═╡ a3759217-e22a-429c-9d1e-1db5c5d20a25
md"""
The noise vector $\eta$ in this case isn't precisely the optimal residual, but it is close.
"""

# ╔═╡ febf512d-b9d7-4dfe-9014-c4c3bf49ba1e
norm(b̂-Â*(Â\b))/norm(η)

# ╔═╡ 9c3e8a96-dce0-4d83-a5fb-656f08008cbd
md"""
How does our least squares fit with partial data compare to the noise vector?  Not well!
"""

# ╔═╡ 3d2d8921-c038-4675-bb3c-c283dca5b181
norm(b̂-Â*ĉ)/norm(η)

# ╔═╡ dc575029-74e3-43ff-a56a-9699564d17c5
md"""
This is not great, though still better than our worst-case error bounds.
"""

# ╔═╡ b2ec63d6-8922-4ad3-88a3-8fea2173332d
begin
	U1, σ1s, V1 = svd(Â)
	1.0 + norm(A)/σ1s[end]
end

# ╔═╡ 9d724537-875e-415f-9bee-969f150e2386
md"""
Let's now consider the regularization approaches from class.
"""

# ╔═╡ 5c6a23cb-25b5-4ae4-907f-07acb9bddb4b
md"""
## Pivoted QR

The pivoted QR factorization $\hat{A}_1 \Pi = QR$ tends to pull "important factors" to the front.  In this case, we can see from the structure of $R$ that columns 1 and 4 are nearly within the span of columns 3 and 2 (to within a residual of around $10^{-12}$).  Therefore, we consider fitting a model that depends just on columns 3 and 2.
"""

# ╔═╡ ee2583bd-6519-4224-928f-fec89ca343d5
F = qr(Â1, ColumnNorm())

# ╔═╡ 646deba6-96c7-47e9-98d3-75e46f4aaf60
norm(Â1[:,F.p]-F.Q*F.R)   # Sanity check that this does what I said it would!

# ╔═╡ 0bac6d3d-8334-45f1-83e5-5516c918c98d
begin
	c_pivqr = zeros(4)
	c_pivqr[F.p[1:2]] = F.R[1:2,1:2]\(F.Q'*b̂1)[1:2]
	c_pivqr
end

# ╔═╡ 9102e5ef-ab4c-424d-9c7b-f1e95f72903a
md"""
This model gets pretty close to the best we could expect!
"""

# ╔═╡ 51a13d05-1cbe-4df7-9057-60df08fcd73d
norm(b̂-Â*c_pivqr)/norm(η)

# ╔═╡ 90ea7cf7-aa0e-44b0-b17c-2ce0de35f2c9
md"""
## Tikhonov regularization

Recall from class that the Tikhonov approach can be encoded as an ordinary least squares problem.  For example, to minimize $\|\hat{A}_1 c - \hat{b}_1\|^2 + 10^{-12} \|c\|^2$ with respect to $c$, we can write
"""

# ╔═╡ e88d7268-3915-4669-802f-a67a039a2107
c_tik1 = [Â1; 1e-6*I]\[b̂1; zeros(4)]

# ╔═╡ 788b0b96-d2f3-473d-bd69-e8e81d48987b
md"""
Indeed, this works pretty well, though you should wonder "where did $10^{-12}$ come from?"
"""

# ╔═╡ d37b6bd7-0e52-42d8-81b2-6d979da1d727
norm(b̂-Â*c_tik1)/norm(η)

# ╔═╡ a0a6bd14-efed-47e1-aa76-feb363f657f9
md"""
The Tikhonov regularized problem can also be expressed via the SVD, where instead of using the inverse singular values $\sigma_i^{-1}$ we use the regularized version $\sigma_i/(\sigma_i^2+\lambda^2)$.
"""

# ╔═╡ 2acb470f-bbd4-41e3-b2db-85ae696a816e
function fit_tikhonov_svd(A, b, λ)
	U, σ, V = svd(A)
	σ̂inv = σ./(σ.^2 .+ λ^2)
	V*(σ̂inv.*(U'*b))
end

# ╔═╡ c0a4429c-0b18-48a4-8483-d5c52eea04fc
fit_tikhonov_svd(Â1, b̂1, 1e-6)

# ╔═╡ f533d8e5-3d78-489e-bbf8-62bc6da248bb
md"""
The SVD is more expensive to compute with than a QR factorization, and is not "sparsity friendly" in the same way that QR is.  But an advantage of using the SVD is that we can quickly recompute solutions associated with different levels of regularization.
"""

# ╔═╡ 52b5d84d-44e1-4460-a22e-ecf917274590
md"""
## Truncated SVD

The truncated SVD approach involves using the first $k$ singular values and vectors to compute an approximate solution to the least squares problem (vs using all of them).
"""

# ╔═╡ 5b38b5cb-bff2-4418-bebc-93a52e525cc1
function fit_truncated_svd(A, b, k)
	U, σ, V = svd(A)
	σ̂inv = 1.0./σ
	σ̂inv[k+1:end] .= 0.0
	V*(σ̂inv.*(U'*b))
end

# ╔═╡ 03293735-0ab1-4d8f-8856-a908b6177107
c_tsvd = fit_truncated_svd(Â1, b̂1, 2)

# ╔═╡ 26485efd-c74b-40ac-9c2c-d86732e57a24
md"""
Again, this approach works pretty well for this problem.
"""

# ╔═╡ 44b03517-b7a6-494a-bab2-a3ecdea649a7
norm(b̂-Â*c_tsvd)/norm(η)

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
# ╟─0064d806-9d7c-11ec-0a5b-052a00b0c0bb
# ╠═ae5453a7-6d80-4fa3-9475-2320cec847cc
# ╠═d8cfd77b-05fe-4b6f-a78b-8304901f5977
# ╟─ba8c2659-893c-47ba-b194-0f8e9b9992ee
# ╠═928f6a09-f65c-4dbf-bf5c-768e0bcc5c80
# ╟─a3759217-e22a-429c-9d1e-1db5c5d20a25
# ╠═febf512d-b9d7-4dfe-9014-c4c3bf49ba1e
# ╟─9c3e8a96-dce0-4d83-a5fb-656f08008cbd
# ╠═3d2d8921-c038-4675-bb3c-c283dca5b181
# ╟─dc575029-74e3-43ff-a56a-9699564d17c5
# ╠═b2ec63d6-8922-4ad3-88a3-8fea2173332d
# ╟─9d724537-875e-415f-9bee-969f150e2386
# ╟─5c6a23cb-25b5-4ae4-907f-07acb9bddb4b
# ╠═ee2583bd-6519-4224-928f-fec89ca343d5
# ╠═646deba6-96c7-47e9-98d3-75e46f4aaf60
# ╠═0bac6d3d-8334-45f1-83e5-5516c918c98d
# ╟─9102e5ef-ab4c-424d-9c7b-f1e95f72903a
# ╠═51a13d05-1cbe-4df7-9057-60df08fcd73d
# ╟─90ea7cf7-aa0e-44b0-b17c-2ce0de35f2c9
# ╠═e88d7268-3915-4669-802f-a67a039a2107
# ╟─788b0b96-d2f3-473d-bd69-e8e81d48987b
# ╠═d37b6bd7-0e52-42d8-81b2-6d979da1d727
# ╟─a0a6bd14-efed-47e1-aa76-feb363f657f9
# ╠═2acb470f-bbd4-41e3-b2db-85ae696a816e
# ╠═c0a4429c-0b18-48a4-8483-d5c52eea04fc
# ╟─f533d8e5-3d78-489e-bbf8-62bc6da248bb
# ╟─52b5d84d-44e1-4460-a22e-ecf917274590
# ╠═5b38b5cb-bff2-4418-bebc-93a52e525cc1
# ╠═03293735-0ab1-4d8f-8856-a908b6177107
# ╟─26485efd-c74b-40ac-9c2c-d86732e57a24
# ╠═44b03517-b7a6-494a-bab2-a3ecdea649a7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
