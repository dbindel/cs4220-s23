### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 67407319-ab93-402b-b281-67afecac152e
using LinearAlgebra

# ╔═╡ 487c4b1c-7fd9-11ec-11b8-d9640811f522
md"""
# Midterm for CS 4220

You may use any resources (course notes, books, papers) except consulting other people, providing attribution for any good ideas you might get.  Solutions that involve computing an explicit inverse (or pseudoinverse) without directions to do so will *not* get full credit.
"""

# ╔═╡ a56aea24-55dd-474d-b733-cabd18036bc6
md"""
#### 1. Efficient operations (4 points)

Rewrite each of the following codes to have the desired complexity.
"""

# ╔═╡ 08d5fd36-fba0-4e5a-b2b3-8a8ea02318d8
begin
	p1a_ref(A, b) = A^3 * b  # Make O(n^2)

	# Solve c' inv(A) B where A is n-by-n, B is n-by-k, and c is a length n vector
	p1b_ref(c, A, B) = c' * (A \ B)  # Make O(n^2 + nk)

	# Compute ek' * A * b - rewrite to be O(n)
	function p1c_ref(A, k, b)
		ek = zeros(length(b))
		ek[k] = 1
		ek' * A * b
	end

	# Compute determinant of [A b; c' d] when we have A = LU.
	# Here A is n-by-n, b, c are vectors, d scalar
	p1d_ref(R, b, d) = det([R'*R b; b' d])  # Make O(n^2)
end

# ╔═╡ d29eb6a8-a05c-44c4-a2c0-9fc07c277b8d
# ╠═╡ disabled = true
#=╠═╡
# You may want to run this sanity checker!  Just re-enable the cell.
# I used p1a, ... for the non-reference versions of the code.
let
	# Sanity check correctness
	A = rand(10,10)
	A = A'*A
	A = (A+A')/2
	b = rand(10)
	B = rand(10,3)
	c = rand(10)
	d = rand()
	F = cholesky(A)
	R = F.U

	norm(p1a(A,b)-p1a_ref(A,b))/norm(p1a_ref(A,b)),
	norm(p1b(c, A, B)-p1b_ref(c, A, B))/norm(p1b_ref(c, A, B)),
	norm(p1c(A, 5, b)-p1c_ref(A, 5, b))/norm(p1c_ref(A, 5, b)),
	norm(p1d(R, b, d)-p1d_ref(R, b, d))/norm(p1d_ref(R, b, d))
end
  ╠═╡ =#

# ╔═╡ 9405b7f6-2b2f-4726-8fd9-e684af9e95ff
md"""
#### 2. A bit of basis (4 points)

The Chebyshev polynomials are $T_0(x) = 1$, $T_1(x) = x$, and then

$$T_{j+1}(x) = 2x T_j(x) - T_{j-1}(x).$$

The Legendre polynomials are $P_0(x) = 1$, $P_1(x) = x$, and then

$$(n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x).$$

The Legendre polynomials satisfy

$$\int_{-1}^1 P_i(x) P_j(x) \, dx = \begin{cases} \frac{2}{2j+1}, & i = j \\ 0, & \mbox{otherwise} \end{cases}$$

The function `poly_xform` below computes a matrix

$$A = \begin{bmatrix} a_{00} & a_{01} & \ldots & a_{0d} \\
  0 & a_{11} & \ldots & a_{1d} \\
  0 & 0 & \ddots & \vdots \\
  0 & 0 & \ldots & a_{dd} \end{bmatrix}$$

such that

$$T_j(x) = \sum_{i = 0}^j P_i(x) A_{ij}.$$

If $T(x)$ and $P(x)$ are row vectors of Chebyshev and Legendre polynomials, we can write this relation concisely as $T(x) = P(x) A$.
"""

# ╔═╡ fc349c3c-cc0c-408a-84a6-d78c633b075c
# Matrix whose columns are Legendre coefficients for Chebyshev polynomials
function poly_xform(d)
	A = zeros(d+1,d+1)
	if d > 0  A[1,1] = 1.0  end
	if d > 1  A[2,2] = 1.0  end
	for j = 2:d
		A[:,j+1] .= -A[:,j-1]
		A[2,j+1] += 2*A[1,j]
		for k = 2:j
			c = 2*A[k,j]/(2*k-1)
			A[k-1,j+1] += (k-1)*c
			A[k+1,j+1] += k*c
		end
	end
	UpperTriangular(A)
end

# ╔═╡ bf2743f6-e255-4f3e-a9e5-d14dada67c54
md"""
- (1 point) Suppose $p(x) = \sum_{j=0}^d T_j(x) c_j = T(x) c$ and $A$ is given by `poly_xform`.  How can we compute $w$ such that $p(x) = \sum_{j=0}^d P_j(x) w_j = P(x) w$?
- (1 point) Argue that $\int_{-1}^1 p(x)^2 \, dx = \sum_{j=0}^d 2 w_j^2 / (2j+1)$.
- (2 points) It is possible to write $\int_{-1}^1 p(x)^2 \, dx$ as $c^T M c$ for some symmetric positive definite matrix $M$.  How could you compute $M$ using the tools above, and what is $\int_{-1}^1 p(x)^2 \, dx$ for the case given below?
"""

# ╔═╡ 7ffb2592-92df-4bae-a2f3-45ae57bb64d6
let
	# Coefficient vector for example function (polynomial approx for exp(x))
	c = [1.26606587775200818;
	     1.13031820798497007;
	     0.271495339534076507;
	     0.0443368498486638174;
	     0.00547424044209370542;
	     0.000542926311914030628;
	     0.0000449773229543007058;
	     0.00000319843646242419376;
	     0.000000199212480641916156;
	     0.0000000110367716525872165;
	     0.000000000550589632227637161]

	# TODO: Fill in computation of M and integral here
end

# ╔═╡ f9693712-b428-48e2-9745-8f104f80e0d7
md"""
#### 3. Closest point (4 points)

A certain optimization algorithm involves repeatedly solving linear systems of the form

$$\begin{bmatrix} A & c \\ c^T & 0 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f \\ g \end{bmatrix}$$

where $c, f, g \in \mathbb{R}^n$ change at each step, but the symmetric positive definite matrix $A \in \mathbb{R}^{n \times n}$ remains fixed.  Complete the following function to solve the above linear system in $O(n^2)$ time given a precomputed Cholesky factor $R$ of $A$.  For full credit, do it in three solves with $R$ or $R^T$ (and $O(n)$ additional work).

Scoring:
- (1 point) Correct derivation
- (2 point) Working $O(n^2)$ code
- (1 point) Done in three solves
"""

# ╔═╡ 1f6e1d72-805b-40c9-99ff-0d0538d7d2f1
function solve_p2(R :: UpperTriangular, c, f, g)
	# TODO: Solve the linear system above
	# (Comment out the following two lines, just to illustrate)
	u = f
	v = 0
	
	u, v
end

# ╔═╡ 151257d8-7811-4d02-bddf-d2b8a96d5e3f
# A tester for your convenience
let
	# Set up an SPD matrix
	A = rand(10,10)
	A = A'*A
	A = (A+A')/2
	F = cholesky(A)

	# Set up a test problem
	uref = randn(10)
	vref = randn()
	c = rand(10)
	f = A*uref + c*vref
	g = c'*uref

	# Solve and compare with reference
	u, v = solve_p2(F.U, c, f, g)
	norm(u-uref)/norm(uref), norm(v-vref)/norm(vref)
end

# ╔═╡ aeb763b3-8bd0-4bec-89e9-46d0913887d9
md"""
#### 4. Silly structure (4 points)

Let $A \in \mathbb{R}^{n \times n}$ be upper triangular except for the last row.

- (2 points) Write an $O(n^2)$ function to compute an unpivoted LU decomposition of $A$
- (2 points) Write an $O(n^2)$ function to compute $R$ in a QR factorization of $A$.
"""

# ╔═╡ 7793fc54-dc15-4ee6-a804-99c3644739b7
# Compute Householder reflector vector v s.t. 
# Hx = norm(x) e_1, where H = I-2*v*v', norm(v) = 1
function house(x)
	v = copy(x)
	v[1] -= norm(v)
	v /= norm(v)
	v
end

# ╔═╡ 674ea213-47b2-4959-9547-914c63af1ed5
function p4lu(A)
	n = size(A)[1]
	A = copy(A)
	# TODO: Complete this code.
	UnitLowerTriangular(A), UpperTriangular(A)
end

# ╔═╡ 76d1e201-afba-4f7a-8c5b-39d5e5197d34
function p4qr(A)
	n = size(A)[1]
	A = copy(A)
	# TODO: Complete this code
	UpperTriangular(A)
end

# ╔═╡ c3e69765-ed8e-45f8-b91b-7825d3906afb
# A tester for your convenience
let
	# Set up a test matrix; make diagonally dominant to avoid pivoting
	A = rand(5,5)
	A = triu(A)
	A[5,:] = rand(5)
	A = A + 5*I

	F1 = lu(A)
	F2 = qr(A)

	L, U = p4lu(A)
	R = p4qr(A)
	R .*= sign.(diag(R)./diag(F2.R))  # Normalize for sign ambiguity in columns

	norm(F1.L-L)/norm(F1.L), 
	norm(F1.U-U)/norm(F1.U),
	norm(F2.R-R)/norm(F2.R)
end

# ╔═╡ 56bbb1f2-78e8-42d2-99bd-1018be42930c
md"""
#### 5. Fun with floating point (4 points)

Consider the linear system

$$\begin{bmatrix} 1 & 1-c \\ 1-c & 1 \end{bmatrix}
  \begin{bmatrix} x \\ y \end{bmatrix} =
  \begin{bmatrix} f - d \\ f + d \end{bmatrix}$$

We note that inverse of the matrix can be written as

$$\begin{bmatrix} 1 & 1-c \\ 1-c & 1 \end{bmatrix}^{-1} = 
\frac{1}{1-(1-c)^2} \begin{bmatrix} 1 & -(1-c) \\ -(1-c) & 1 \end{bmatrix}$$

where $c, d$ are small positive real numbers of similar magnitude (say around $\epsilon_{\mathrm{mach}}$) and $f$ is around 1.

We initially consider a straightforward solver for this system using Julia's backslash.
"""

# ╔═╡ cd13696f-c9f2-4fb1-9e2e-7eb5926b37c7
p5v1(c, d, f) = [1 1-c; 1-c 1]\[f-d; f+d]

# ╔═╡ 9d0a117e-a43b-48fe-b227-f73e7895b2bc
md"""
- (1 point) Explain the exception the happens with `p5v1(1e-17, 3e-17, 1.5)` (but not with `p5v1(1e-16, 3e-16, 1.5)`).
- (1 point) Explain why `p5v1(1e-16, 3e-16, 1.5)` is inaccurate.
- (2 point) Write `p5solve(c, d, f)` to get high relative accuracy in both $x$ and $y$.  What is the output of `p5solve(1e-16, 3e-16, 1.5)`?  Why do you believe your modified approach?

*Hint*: It is worth sanity checking your algebra by also comparing to `p5v1` for more benign values of $c$ and $d$ (e.g. $c = 0.1$ and $d = 0.3$).  Also, this is a place where it is fine to start from an explicit inverse (though calling `inv` on the matrix directly works no better than using backslash).
"""

# ╔═╡ b4ffb302-d089-411f-a5d0-a837aba17ddf
md"""
#### 6. Norm! (4 points)

Suppose $M \in \mathbb{R}^{n \times n}$ is an invertible matrix.

- (2 point) Argue that $\|v\|_* = \|Mv\|_\infty$ is a vector norm.
- (2 point) Write a short Julia code to compute the associated operator norm.

*Note*: In Julia, `opnorm(A, Inf)` computes the infinity operator norm of a matrix $A$; `norm(A, Inf)` computes the vector infinity norm (i.e.~$\max_{i,j} |a_{ij}|$).
"""

# ╔═╡ e5179fcb-817f-467b-918f-7dd5f05ccee9
md"""
#### 7. Rank one approximation (4 points)

- (1 point) How would you find $x$ to minimize $\|bx-a\|_2^2$ when $a, b \in \mathbb{R}^m$ are given vectors?
- (1 point) Suppose $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^m$ are given.  How would you find $u \in \mathbb{R}^n$ to minimize $\|A-bu^T\|_F^2$?
- (1 point) Suppose $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^m, c \in \mathbb{R}^n$ are given.  Show that $\langle A, bc^T \rangle_F = b^T A c$.  (Recall that the Frobenius inner product is $\langle X, Y \rangle_F = \sum_{i,j} y_{ij} x_{ij} = \operatorname{tr}(Y^T X)$).
- (1 point) Suppose $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^m$ and $c \in \mathbb{R^n}$ are given.  How would you find $\gamma \in \mathbb{R}$ to minimize $\|A-\gamma bc^T\|_F^2$?
"""

# ╔═╡ aa9b8535-f460-49de-a8ec-ed7508e89613
md"""
#### 8. Your turn (2 points)

Choose your favorite two, or answer all three!

- What is one thing you think is going particularly well in the class?
- What is one thing you think could be improved?
- What is one something you find particularly interesting right now, from this class or otherwise?
"""

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
# ╟─487c4b1c-7fd9-11ec-11b8-d9640811f522
# ╠═67407319-ab93-402b-b281-67afecac152e
# ╟─a56aea24-55dd-474d-b733-cabd18036bc6
# ╠═08d5fd36-fba0-4e5a-b2b3-8a8ea02318d8
# ╠═d29eb6a8-a05c-44c4-a2c0-9fc07c277b8d
# ╟─9405b7f6-2b2f-4726-8fd9-e684af9e95ff
# ╠═fc349c3c-cc0c-408a-84a6-d78c633b075c
# ╟─bf2743f6-e255-4f3e-a9e5-d14dada67c54
# ╠═7ffb2592-92df-4bae-a2f3-45ae57bb64d6
# ╟─f9693712-b428-48e2-9745-8f104f80e0d7
# ╠═1f6e1d72-805b-40c9-99ff-0d0538d7d2f1
# ╠═151257d8-7811-4d02-bddf-d2b8a96d5e3f
# ╟─aeb763b3-8bd0-4bec-89e9-46d0913887d9
# ╠═7793fc54-dc15-4ee6-a804-99c3644739b7
# ╠═674ea213-47b2-4959-9547-914c63af1ed5
# ╠═76d1e201-afba-4f7a-8c5b-39d5e5197d34
# ╠═c3e69765-ed8e-45f8-b91b-7825d3906afb
# ╟─56bbb1f2-78e8-42d2-99bd-1018be42930c
# ╠═cd13696f-c9f2-4fb1-9e2e-7eb5926b37c7
# ╟─9d0a117e-a43b-48fe-b227-f73e7895b2bc
# ╟─b4ffb302-d089-411f-a5d0-a837aba17ddf
# ╟─e5179fcb-817f-467b-918f-7dd5f05ccee9
# ╟─aa9b8535-f460-49de-a8ec-ed7508e89613
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
