### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 67407319-ab93-402b-b281-67afecac152e
using LinearAlgebra

# ╔═╡ dfa9a07e-a7ea-4a38-97f7-9854ad6d6fe9
using BenchmarkTools

# ╔═╡ 487c4b1c-7fd9-11ec-11b8-d9640811f522
md"""
# HW 1 for CS 4220

You may (and probably should) talk about problems with the each other, with the TAs, and with me, providing attribution for any good ideas you might get.  Your final write-up should be your own.
"""

# ╔═╡ 5fd85ce3-d746-4682-b9ce-8980b6692a3c
md"""
#### 1: Placing Parens

Suppose $A, B \in \mathbb{R}^{n \times n}$ and $d, u, v, w \in \mathbb{R}^n$.  Rewrite each of the following Julia functions to compute the same result but with the indicated asymptotic complexity.
"""

# ╔═╡ fcd2b4ff-e6f8-4c94-8f2a-46b0dd239005
# Rewrite to run in O(n)
function hw1p1a(A, d)
	D = diagm(d)
	tr(D*A*D)
end

# ╔═╡ 2fff33f2-d129-41ae-a949-5946e534019a
begin
	function test_hw1p1a()
		A = [1.0 2.0; 3.0 4.0]
		d = [9.0; 11.0]
		hw1p1a(A, d) == 565.0
	end
	
	if test_hw1p1a()
		"P1a code passes correctness test"
	else
		"P1a code fails correctness test"
	end
end

# ╔═╡ d7feb833-2e95-4272-b87d-21b2db67872f
# Rewrite to run in O(n^2)
function hw1p1b(A, B, u, v)
	C = u*v'
	A*C*B
end

# ╔═╡ 3c26307e-71ca-40c5-aa11-4073bfd31fd4
begin
	function test_hw1p1b()
		A = [9.0 15.0; 11.0 16.0]
		B = [8.0 17.0; 12.0 18.0]
		u = [7.0; 11.0]
		v = [4.0; 5.0]
		hw1p1b(A, B, u, v) == [20976.0 36024.0; 23276.0 39974.0]
	end
	
	if test_hw1p1b()
		"P1b code passes correctness test"
	else
		"P1b code fails correctness test"
	end
end

# ╔═╡ dc1e6d3d-e205-429c-a47f-0d144fc25a09
# Rewrite to run in O(n^2)
function hw1p1c(A, B, d, w)
	(diagm(d) + A*B)*w
end

# ╔═╡ f3efa1d2-7c2e-4879-b8d1-9e424bc098bf
begin
	function test_hw1p1c()
		A = [1.0 2.0; 3.0 4.0]
		B = [5.0 6.0; 7.0 8.0]
		d = [8.0; 12.0]
		w = [4.0; 5.0]
		hw1p1c(A, B, d, w) == [218.0, 482.0]
	end
	if test_hw1p1c()
		"P1c code passes correctness test"
	else
		"P1c code fails correctness test"
	end
end

# ╔═╡ b4407107-057e-4c6d-9308-0aa12766a755
md"""
#### 2: Making matrices

Recall the [first kind Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials), which satisfy

$$\begin{align*}
  T_0(x) &= 1 \\
  T_1(x) &= x \\
  T_{n+1}(x) &= 2x T_n(x) - T_{n-1}(x), \quad n \geq 1.
\end{align*}$$

1.  Write the matrix $A \in \mathbb{R}^{5 \times 5}$ representing the linear map from the coefficients of $p \in \mathcal{P}_4$ in the Chebyshev basis to the coefficients of the same $p \in \mathcal{P}_4$ in the power basis.

2.  Write the matrix $B \in \mathbb{R}^{5 \times 4}$ representing the linear map $p(x) \mapsto x p(x)$ from $\mathcal{P}_3$ to $\mathcal{P}_4$ with respect to the Chebyshev basis for both spaces.

*Hint for 2*: Observe that the Chebyshev recurrence can also be written as

$$x T_n(x) = \frac{1}{2} T_{n-1}(x) + \frac{1}{2} T_{n+1}(x)$$
"""

# ╔═╡ 02872ae8-e7c4-45b6-a386-a97a5c3ef4dd
md"""
#### 3: Crafty cosines

Suppose $\| \cdot \|$ is a inner product norm in some real vector space and you are given

$$a = \|u\|, \quad b = \|v\|, \quad c = \|u-v\|$$

Express $\langle u, v \rangle$ in terms of $a$, $b$, and $c$.  Be sure to explain where your formula comes from.
"""

# ╔═╡ 0f70e29d-76f4-483b-8ce3-af031eb987ab
function compute_dot(a, b, c)
	return 0.0
end

# ╔═╡ 03d1dfa5-aa1c-4222-8174-abdee1bf1557
begin
	function test_dot_abc()
		u = rand(10)
		v = rand(10)
		a = norm(u)
		b = norm(v)
		c = norm(u-v)
		d1 = compute_dot(a, b, c)
		d2 = v'*u
		(d2-d1)/d2
	end
	
	if abs(test_dot_abc()) < 1e-8
		"Passes sanity check"
	else
		"Fails sanity check"
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[compat]
BenchmarkTools = "~1.2.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "ab4991bc3efa10e31ffe65a43b739c59fc039b8a"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "940001114a0147b6e4d10624276d56d531dd9b49"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0b5cfbb704034b5b4c1869e36634438a047df065"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"
"""

# ╔═╡ Cell order:
# ╟─487c4b1c-7fd9-11ec-11b8-d9640811f522
# ╠═67407319-ab93-402b-b281-67afecac152e
# ╠═dfa9a07e-a7ea-4a38-97f7-9854ad6d6fe9
# ╟─5fd85ce3-d746-4682-b9ce-8980b6692a3c
# ╠═fcd2b4ff-e6f8-4c94-8f2a-46b0dd239005
# ╟─2fff33f2-d129-41ae-a949-5946e534019a
# ╠═d7feb833-2e95-4272-b87d-21b2db67872f
# ╟─3c26307e-71ca-40c5-aa11-4073bfd31fd4
# ╠═dc1e6d3d-e205-429c-a47f-0d144fc25a09
# ╟─f3efa1d2-7c2e-4879-b8d1-9e424bc098bf
# ╟─b4407107-057e-4c6d-9308-0aa12766a755
# ╟─02872ae8-e7c4-45b6-a386-a97a5c3ef4dd
# ╠═0f70e29d-76f4-483b-8ce3-af031eb987ab
# ╟─03d1dfa5-aa1c-4222-8174-abdee1bf1557
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
