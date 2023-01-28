### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ ae26058a-0f9c-4572-9808-05d3c7b2ec64
using LinearAlgebra

# ╔═╡ 9b69b962-4973-4608-b5d5-ee539ff06672
md"""
# Notebook for 2023-02-03
"""

# ╔═╡ d9dcb851-36d1-4081-b73e-30d8d93ba3bf
md"""
## Representations

### Normalized numbers

In the notes, we said 1/3 as a normalized binary number looks like

$$(-1)^{0} \times (1.010101...)_2 \times 2^{-2}$$

The bit string for this number involves:

- A leading sign bit (which is 0 for a positive number)
- An 11-bit exponent (`01111111101`) that represents the true exponent + 1023.  The cases where all the bits are zero or are all one represent subnormals and infinity/NaN representations, respectively.
- The remaining 52 bits `0101010101010101010101010101010101010101010101010101` represent the significant *after* the binary point.  We don't need to store the leading digit, because it is always one for a normalized binary number.
"""

# ╔═╡ 8d4e3f0e-92c4-11ec-2761-c306dbe9e6c3
bitstring(1.0/3.0)

# ╔═╡ a103e7c4-9a25-4516-8c6f-e06f494b5cc3
bitstring(1.0/3.0)[1]  # Sign big

# ╔═╡ 1442c5b7-e0a0-46f6-8807-032aba6caeca
bitstring(1.0/3.0)[2:12]  # Exponent bits

# ╔═╡ eee31e78-2589-4834-b1c4-25a06d2253c2
bitstring(1.0/3.0)[13:end]  # Significand bits after the binary point

# ╔═╡ 5b0023a8-47fb-4525-843a-838eb19fb84a
md"""
### Subnormals

The subnormal numbers (64-bit format) look like

$$(-1)^s \times (0.b_1 b_2 \ldots b_{52})_2 \times 2^{-1023}$$

where $s$ is the sign bit, $b_1$ through $b_52$ are the significand bits, and the exponent of $-1023$ is represented by the all-zeros bit pattern in the exponent field.  A good nonzero example is $2^-1025$.
"""

# ╔═╡ 07c689e3-ab3c-4d42-8fd0-78a3e90c3390
bitstring(2.0^-1025)

# ╔═╡ b92c5018-01e0-4f53-a0f6-0b759338a2f6
bitstring(2.0^-1025)[2:12]

# ╔═╡ de5311d5-8c85-4ffe-a2e9-4defe051d84a
bitstring(2.0^-1025)[13:end]

# ╔═╡ 6687a352-3172-48a0-a690-e92f9b81f17a
md"""
A number like 0 is subnormal, so all the exponent bits (as well as all the other bits) will be zero.
"""

# ╔═╡ 59bdfcbe-3ca6-4dd4-9b84-66a2d20ad4a7
bitstring(0.0)

# ╔═╡ 026f0aee-76cc-4139-9ec1-c874f61fbc77
md"""
We do have the possibility of -0, which has a one for the sign bit (and everything else zero).
"""

# ╔═╡ af1bdbe3-78c2-49e7-92f2-5b4a69deac43
bitstring(-0.0)

# ╔═╡ 3a49b8a2-6026-4980-9c0f-6594f32ad0da
md"""
### Infinities

Infinities are represented by the all-ones exponent pattern and a significand field which is all zeros.
"""

# ╔═╡ 9c60cafc-fbd9-4b7d-a560-00f717ea6962
bitstring(Inf)

# ╔═╡ e09d16d0-2832-4eea-b47f-49a72c397ab3
bitstring(Inf)[2:12]  # Exponent bits

# ╔═╡ 83106e79-b940-4b15-a0c1-d478d2aee38c
bitstring(Inf)[13:end]  # Significand bits

# ╔═╡ cf31a79a-0e3b-4aaf-a5ba-645283dff785
md"""
### NaNs

Not-a-Number (NaN) representations share the same all-ones exponent pattern, but can encode additional data in the (nonzero) significand bits.
"""

# ╔═╡ 2ada806a-fa22-4737-b2fd-2b10fc10700a
bitstring(0.0/0.0)

# ╔═╡ e80fd10c-8ff7-45ba-8bd2-1ac9b692d131
bitstring(0.0/0.0)[13:end]

# ╔═╡ 11d0646f-5bf2-4a20-93a3-512340cc446b
md"""
## Illustrated floating point mishaps
"""

# ╔═╡ 617f29d4-94ea-42b7-a3a7-db38afdefd88
md"""
### Cancellation

The standard example here is the smaller root of a quadratic

$$z^2 - 2z + \epsilon = 0$$

Thinking of the smaller root $z_-$ as a function of $\epsilon$, we can use implicit differentiation to find

$$2(z-1) \frac{dz}{d\epsilon} + 1 = 0$$

which means that near zero, we expect

$$z_- = \frac{\epsilon}{2} + O(\epsilon^2)$$

Unfortunately, the naive formula utterly fails us for small $\epsilon$.
"""

# ╔═╡ 87969ad1-8611-4190-b101-de41a6652aef
test_smaller_root(ϵ) = 1.0 - sqrt(1.0 - ϵ)

# ╔═╡ 778892f5-69d0-437c-ad99-8df4c3791ab9
test_smaller_root(1e-10)

# ╔═╡ 87056b10-95cd-466c-b717-5f6e3448914c
test_smaller_root(1e-15)

# ╔═╡ ec76198e-12bd-4e65-8cea-785071c1c379
test_smaller_root(1e-20)

# ╔═╡ 62972063-a924-4505-9d6e-474fb7dfc652
md"""
The problem here is cancellation: error committed in computing $1-\epsilon$ is small relative to that quantity (which is around 1), but is big relative to the size of $z_-$ (which is wround $\epsilon/2$).  An alternate formulation (the product divided by the larger root) does not involve subtracting things that are very close, and so does not suffer from amplification of error due to cancellation.
"""

# ╔═╡ c4c4eef3-5149-48ab-ab18-7b27e733d073
test_smaller_root2(ϵ) = ϵ/(1.0 + sqrt(1.0 - ϵ))

# ╔═╡ c57a26ce-7c1b-43ad-91e1-9b66d9899ad3
test_smaller_root2(1e-10)

# ╔═╡ 56698842-32dd-4d70-8f98-be2b188a23a2
test_smaller_root2(1e-15)

# ╔═╡ a050a231-791c-4f66-9e7e-996b1c3a68cc
test_smaller_root2(1e-20)

# ╔═╡ 28495ea1-946d-45b7-bd1d-7791e303dd2f
md"""
### Sensitive subproblems

In the notes, we described the case of taking many square roots followed by many squares.  The problem is that the problem of taking many squares is extremely ill-conditioned, so that even a relative error of machine epsilon in the result of the first loop can lead to a very large error in the result of the second loop (even assuming no further rounding error -- as is the case when the first loop computes exactly 1.0 in floating point!).
"""

# ╔═╡ 371312cd-941a-4161-ba38-164b0060e15c
function silly_sqrt(n=100)
	x = 2.0
	for k = 1:n
		x = sqrt(x)
	end
	for k = 1:n
		x = x^2
	end
	x
end

# ╔═╡ b9095dfa-107d-4a66-ade4-8ba6952c61eb
silly_sqrt(10)

# ╔═╡ 04f16ecf-a3e5-4339-9ac8-9a35d977bce7
silly_sqrt(20)

# ╔═╡ 3608450e-c794-4efa-8cbe-e313c21c7adb
silly_sqrt(40)

# ╔═╡ 40ca6288-3c2b-4e22-87e6-61215eeceec0
silly_sqrt(60)

# ╔═╡ 82c03d9f-7c99-4455-9a76-cc6013ecf6b6
md"""
### Exploding recurrences

Consider the computation of

$$E_n = \int_0^1 x^n e^{-x} \, dx$$

By inspection, we know that $0 \leq E_n \leq E_0 = 1-1/e$ for $n \geq 0$.  An asymptotic argument gives us that for large $n$, we should have $E_n \approx 1/(e(n+1))$.  And integration by parts gives the recurrence

$$E_n = 1-nE_{n-1}.$$

Unfortunately, if we run this recurrence forward, the error at each step gets multiplied by a factor of $n$, and things soon become hopelessly awful.
"""

# ╔═╡ a71743fd-5958-418e-9901-e0c0c5242576
function bad_recurrence(n)
	E = 1-1/exp(1)
	for j = 1:n
		E = 1-j*E
	end
	E
end

# ╔═╡ d19bf7dd-10ae-43c2-9f7c-6f4f167ec51f
bad_recurrence(1)

# ╔═╡ 7e1f4f16-0061-46a4-b08d-1096b04f493d
bad_recurrence(20)

# ╔═╡ 8a77538e-0774-48eb-9884-839f3c1c8c83
md"""
A better approach is to run the same recurrence backward from our (crude) estimate.
"""

# ╔═╡ 113d9515-a463-416e-b5f2-075e8d10669c

function better_recurrence(n)
	E = 1.0/exp(1.0)/(n+100)
	for j = n+100:-1:n
		E = (1-E)/(j+1)
	end
	E
end

# ╔═╡ d2fa121d-e2cb-4a9d-8391-6d0a75f64ce3
better_recurrence(1)

# ╔═╡ a6b39980-cfbd-43ab-b2d4-af5c2278a7fe
better_recurrence(20)

# ╔═╡ 0fb54f06-af42-44b8-8a60-b0e8b94ed6e1
md"""
### Undetected underflow

The case we mentioned in the notes comes from Bayesian statistics.  Suppose we want to compute the log determinant of a large matrix -- we'll choose something simple.
"""

# ╔═╡ db6cde6b-6969-4f6c-848b-ae2992a9a17d
D = Diagonal(0.05*ones(500))

# ╔═╡ cd8e25d5-74ab-41f3-9fc3-a8995068b51e
md"""
If we use the obvious formula, the product while underflow, leading us to take the log of zero.
"""

# ╔═╡ 8475c362-dc50-4dc4-a027-2f38eac23631
log(det(D))

# ╔═╡ a6cbeff8-5630-42e6-960a-585b3c0ca64e
md"""
Of course, this is not the right answer!  If we recognize that the log of a product is the sum of the logs, we can easily figure out the true log determinant in this case.
"""

# ╔═╡ 7c9a9698-e61f-4899-8608-4c65cbb70264
500*log(0.05)

# ╔═╡ 014bc10f-6c60-4c1b-9c8b-1cd4f091d598
md"""
### Bad branches

The key point here is that NaN fails all comparisons -- it is not part of the usual ordering relations, and violates our expectations.  Therefore, we can get into trouble with branches when NaNs are involved.
"""

# ╔═╡ c7cc0fb5-7b7f-4e88-9cc2-e2f0a7728956
function test_negative(x)
	if x < 0.0
		"$(x) is negative"
	elseif x >= 0.0
		"$(x) is non-negative"
	else
		"$(x) is ... uh..."
	end
end

# ╔═╡ b81cb74d-6a04-452e-a500-444f72a0464a
test_negative(1.0)

# ╔═╡ bffcadfa-5c37-495a-8257-33e8a8823d6f
test_negative(-1.0)

# ╔═╡ 182dd044-72e8-46fb-9f66-570316482c1b
test_negative(0.0/0.0)

# ╔═╡ 661d3f16-3d4e-4e49-8c42-dbf53b71b201
md"""
Of course, there are some other subtleties here, too!  For example, the floating point standard contains both positive and negative zeros, but this code will not distinguish between them (they are the same according to all comparison operations).
"""

# ╔═╡ 7cf8d7af-2fd2-43a1-970c-466ca7b96269
test_negative(-0.0)

# ╔═╡ c2de18e9-2672-4543-a40b-a2349cedf4a0
test_negative(0.0)

# ╔═╡ a7f6b3dd-f8dd-4b6b-bd12-19ceacbc960d
copysign(1.0, -0.0)

# ╔═╡ c06bd4f5-8afc-48b5-884e-82e0e95793bb
copysign(1.0, 0.0)

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
# ╟─9b69b962-4973-4608-b5d5-ee539ff06672
# ╟─d9dcb851-36d1-4081-b73e-30d8d93ba3bf
# ╠═8d4e3f0e-92c4-11ec-2761-c306dbe9e6c3
# ╠═a103e7c4-9a25-4516-8c6f-e06f494b5cc3
# ╠═1442c5b7-e0a0-46f6-8807-032aba6caeca
# ╠═eee31e78-2589-4834-b1c4-25a06d2253c2
# ╟─5b0023a8-47fb-4525-843a-838eb19fb84a
# ╠═07c689e3-ab3c-4d42-8fd0-78a3e90c3390
# ╠═b92c5018-01e0-4f53-a0f6-0b759338a2f6
# ╠═de5311d5-8c85-4ffe-a2e9-4defe051d84a
# ╟─6687a352-3172-48a0-a690-e92f9b81f17a
# ╠═59bdfcbe-3ca6-4dd4-9b84-66a2d20ad4a7
# ╟─026f0aee-76cc-4139-9ec1-c874f61fbc77
# ╠═af1bdbe3-78c2-49e7-92f2-5b4a69deac43
# ╟─3a49b8a2-6026-4980-9c0f-6594f32ad0da
# ╠═9c60cafc-fbd9-4b7d-a560-00f717ea6962
# ╠═e09d16d0-2832-4eea-b47f-49a72c397ab3
# ╠═83106e79-b940-4b15-a0c1-d478d2aee38c
# ╟─cf31a79a-0e3b-4aaf-a5ba-645283dff785
# ╠═2ada806a-fa22-4737-b2fd-2b10fc10700a
# ╠═e80fd10c-8ff7-45ba-8bd2-1ac9b692d131
# ╟─11d0646f-5bf2-4a20-93a3-512340cc446b
# ╟─617f29d4-94ea-42b7-a3a7-db38afdefd88
# ╠═87969ad1-8611-4190-b101-de41a6652aef
# ╠═778892f5-69d0-437c-ad99-8df4c3791ab9
# ╠═87056b10-95cd-466c-b717-5f6e3448914c
# ╠═ec76198e-12bd-4e65-8cea-785071c1c379
# ╟─62972063-a924-4505-9d6e-474fb7dfc652
# ╠═c4c4eef3-5149-48ab-ab18-7b27e733d073
# ╠═c57a26ce-7c1b-43ad-91e1-9b66d9899ad3
# ╠═56698842-32dd-4d70-8f98-be2b188a23a2
# ╠═a050a231-791c-4f66-9e7e-996b1c3a68cc
# ╟─28495ea1-946d-45b7-bd1d-7791e303dd2f
# ╠═371312cd-941a-4161-ba38-164b0060e15c
# ╠═b9095dfa-107d-4a66-ade4-8ba6952c61eb
# ╠═04f16ecf-a3e5-4339-9ac8-9a35d977bce7
# ╠═3608450e-c794-4efa-8cbe-e313c21c7adb
# ╠═40ca6288-3c2b-4e22-87e6-61215eeceec0
# ╟─82c03d9f-7c99-4455-9a76-cc6013ecf6b6
# ╠═a71743fd-5958-418e-9901-e0c0c5242576
# ╠═d19bf7dd-10ae-43c2-9f7c-6f4f167ec51f
# ╠═7e1f4f16-0061-46a4-b08d-1096b04f493d
# ╟─8a77538e-0774-48eb-9884-839f3c1c8c83
# ╠═113d9515-a463-416e-b5f2-075e8d10669c
# ╠═d2fa121d-e2cb-4a9d-8391-6d0a75f64ce3
# ╠═a6b39980-cfbd-43ab-b2d4-af5c2278a7fe
# ╟─0fb54f06-af42-44b8-8a60-b0e8b94ed6e1
# ╠═ae26058a-0f9c-4572-9808-05d3c7b2ec64
# ╠═db6cde6b-6969-4f6c-848b-ae2992a9a17d
# ╟─cd8e25d5-74ab-41f3-9fc3-a8995068b51e
# ╠═8475c362-dc50-4dc4-a027-2f38eac23631
# ╟─a6cbeff8-5630-42e6-960a-585b3c0ca64e
# ╠═7c9a9698-e61f-4899-8608-4c65cbb70264
# ╟─014bc10f-6c60-4c1b-9c8b-1cd4f091d598
# ╠═c7cc0fb5-7b7f-4e88-9cc2-e2f0a7728956
# ╠═b81cb74d-6a04-452e-a500-444f72a0464a
# ╠═bffcadfa-5c37-495a-8257-33e8a8823d6f
# ╠═182dd044-72e8-46fb-9f66-570316482c1b
# ╟─661d3f16-3d4e-4e49-8c42-dbf53b71b201
# ╠═7cf8d7af-2fd2-43a1-970c-466ca7b96269
# ╠═c2de18e9-2672-4543-a40b-a2349cedf4a0
# ╠═a7f6b3dd-f8dd-4b6b-bd12-19ceacbc960d
# ╠═c06bd4f5-8afc-48b5-884e-82e0e95793bb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
