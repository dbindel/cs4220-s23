### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ f44f2491-52ef-4b43-920a-666eac83368b
using LinearAlgebra

# ╔═╡ 88ccddfb-9645-4905-b6f9-8f7a60519e5f
using Plots

# ╔═╡ c2fc87e6-b506-11ec-2093-c5e4c97aae19
md"""
# Notebook for 2023-04-14
"""

# ╔═╡ 0627e5e4-452f-4778-948b-fdaa4c3f60ec
md"""
## Life beyond fixed points

So far, the methods we have discussed for solving nonlinear systems all involve some flavor of fixed point iteration

$$x^{k+1} = G(x^k)$$

Our chief example of such an iteration is Newton’s method, where

$$G(x) = x - f'(x)^{-1} f(x)$$

but we have also considered various other iterations where the Jacobian is replaced by some more convenient approximation. However, all the methods we have discussed in this setting compute the next point based only on the behavior at the current point, and not any previous points.

In earlier parts of the class, of course, we did consider iterations where the next approximation depends on more than just the current approximation. Two such iterations spring particularly to mind:

- The secant method is the 1D iteration in which we approximate the derivative at the current point by a finite difference through the last two points.

- In iterative methods for solving linear systems, we started with fixed point iterations based on matrix splittings (such as Jacobi and Gauss-Seidel), but then recommended them primarily as an approach to preconditioning the more powerful Krylov subspace methods. One way to think about (preconditioned) Krylov subspaces is that they are spanned by the iterates of one of these stationary methods. Taking a general element of this space — a linear combination of the iterates of the stationary method — improves the convergence of stationary methods.

In this lecture, we describe nonlinear analogues to both of these approaches. The generalization of the secant condition to more than one direction gives Broyden’s method, the most popular of the quasi-Newton methods that couple a Newton-like update of the approximate solution with an update for an approximate Jacobian. The generalization of the Krylov subspace approach leads us to extrapolation methods, and we will describe a particular example known as reduced rank extrapolation. We conclude the lecture with a brief discussion of Anderson acceleration, a method that can be seen as a generalization either of the secant approach or of Krylov subspace methods.

As with last time, we will continue to use the autocatalytic reaction-diffusion equation as a running example.
"""

# ╔═╡ 704f1c07-b6b5-4c03-abbc-1951e37b6a7b
begin
	# Take the initial guess from last lecture
	N = 100
	xx = range(1, N, length=N)/(N+1)
	q = xx.*(1.0 .- xx)
	v0 = 0.5*q
end

# ╔═╡ 372c511c-1a35-44fb-958c-ec51a0a034e6
function autocatalytic(v)
    N = length(v)
    fv        = exp.(v)
    fv        -= 2*(N+1)^2*v
    fv[1:N-1] += (N+1)^2*v[2:N  ]
    fv[2:N  ] += (N+1)^2*v[1:N-1]
    fv
end

# ╔═╡ c62b0f6e-8709-4dc5-bc5f-3978b37f6b5c
function Jautocatalytic(v)
    N = length(v)
    SymTridiagonal(exp.(v) .- 2*(N+1)^2, (N+1)^2 * ones(N-1))
end

# ╔═╡ 821e8d55-022e-44c9-b22d-a6eb42b48025
md"""
## Broyden

Quasi-Newton methods take the form

$$x^{k+1} = x^k - J_k^{-1} f(x^k)$$

together with an updating formula for computing successive approximate Jacobians $J_k$. By far the most popular such updating formula is Broyden’s (good) update:

$$J_{k+1} = J_k + \frac{f(x^k) s^k}{\|s^k\|^2}, \quad s^k = x^{k+1}-x^k$$

Broyden’s update satisfies the secant condition

$$J_{k+1} (x^{k+1}-x^k) = f(x^{k+1})-f(x^k)$$

which is the natural generalization of the 1D secant condition. In one dimension, the secant condition is enough to uniquely determine the Jacobian approximation, but this is not the case in more than one dimension. Hence, Broyden’s update gives the matrix $J_{k+1}$ that minimizes $\|J_{k+1}-J_k\|_F$ subject to the secant condition.
"""

# ╔═╡ 7a932860-d0c6-4dfb-9438-1451b8f66cfb
function naive_broyden(x, f, J; rtol=1e-6, nsteps=100, monitor=(x, fx) -> nothing)

    x = copy(x)
    J = Array{Float64}(J)  # Dense matrix representation of the initial Jacobian

    # Take an initial step to get things started
    fx = f(x)
    monitor(x, fx)
    s = -J\fx
    
    for step = 1:nsteps

        # Take Broyden step and update Jacobian (overwrite old arrays)
        x[:] += s
        fx[:] = f(x)
        J[:,:] += fx*(s/norm(s)^2)'
        s[:] = -J\fx

        # Monitor progress and check for convergence
        monitor(x, fx)
        if norm(fx) < rtol
            return x, fx
        end

    end
    
    # We could error out here, but let's just return x and fx
    return x, fx
end

# ╔═╡ bc73550e-29ab-4696-b0a5-57bf2d195d2f
md"""
Broyden’s method has three particularly appealing features:

- For sufficiently good starting points $x^0$ (and sufficiently innocuous initial Jacobian approximations), Broyden’s method is $q$-superlinearly convergent, i.e. there is some constant $K$ and some $\alpha > 1$ such that $\|e^{k+1}\| \leq \|e^k\|^\alpha$.

- The iteration requires only function values. There is no need for analytical derivatives.

- Because each step is a rank one update, we can use linear algebra tricks to avoid the cost of a full factorization at each step that would normally be required for a Newton-type iteration.

The argument that Broyden converges superlinearly is somewhat complex, and we will not cover it in this course; for details of the argument, I suggest Tim Kelley’s [Iterative Methods for Linear and Nonlinear Equations](https://doi.org/10.1137/1.9781611970944).  The fact that the method requires only function values is obvious from the form of the updates. But the linear algebra tricks bear some further investigation, in part because different tricks are used depending on the type of problem involved.
"""

# ╔═╡ d171b1ba-9bb1-4aee-8879-772383b8df3c
let
	rhist = []
	J0 = Jautocatalytic(v0)
	v = naive_broyden(v0, autocatalytic, J0, rtol=1e-10, monitor=(x, fx)->push!(rhist, norm(fx)))
	plot(rhist, yscale=:log10, legend=false)
end

# ╔═╡ 184743b7-8c56-40f7-be96-ad35ff2bb46d
md"""
#### Questions

Give an argument for the claim above that Broyden's update gives the matrix $J_{k+1}$ that minimizes $\|J_{k+1}-J_k\|_F^2$ subject to the secant condition.
"""

# ╔═╡ 4051701e-e695-4b2b-8a5a-80b076615e04
md"""
### Dense Broyden

For moderate-sized problems, implementations of Broyden’s method may maintain a dense factorization of the approximate Jacobian which is updated at each step. This updating can be done economically by exploiting the fact that the update at each step is rank one. For example, the QR factorization of $J_k$ can be updated to a QR factorization of $J_{k+1}$ in $O(n^2)$ time by using a sequence of Givens rotations. More generally, we can compute the QR factorization of the rank one update $A' = A+uv^T$ from the QR factorization $A = \bar{Q} \bar{R}$ in three steps:

1. Write $A' = \bar{Q} (\bar{R} + \bar{u} v^T)$ where $\bar{u} = \bar{Q}^T u$

2. Apply $n-1$ Givens rotations to $\bar{u}$ to introduce zeros in positions $n, n-1, \ldots, 2$.  Apply those same rotations to the corresponding rows of $R$, transformint it into an upper Hessenberg matrix $H$; and apply the transposed rotations to the corresponding columns of $\bar{Q}$ to get $A' = \tilde{Q} (H + \gamma e_1 v^T)$.

3. Do Hessenberg QR on $H + \gamma e_1v^T$ to finish the decomposition, i.e. apply Givens rotations to zero out each subdiagonal in turn.

In total, this computation involves computing $\bar{u}$ (in $O(n^2)$ time) and applying $2n-2$ Givens rotations across the two factors (also a total of $O(n^2$ time).  So updating the factorization of $J_k$ to get a factorization of $J_{k+1}$ takes $O(n^2)$; and once we have a factorization of $J_{k+1}$, linear solves can be computed in $O(n^2)$ time as well.  So the total cost per Broyden step is $O(n^2)$ (after an initial $O(n^3)$ factorization).
"""

# ╔═╡ 40d528bc-596a-4e2d-ae0a-796bab25d680
function qrupdate!(Q, R, u, v)
    n = length(u)
    ubar = Q'*u
    for j = n:-1:2
        G, r = givens(ubar[j-1], ubar[j], j-1, j)
        ubar[j] = 0.
        ubar[j-1] = r
        R[:,j-1:end] = G*R[:,j-1:end]
        Q[:,:] = Q[:,:]*G'
    end
    R[1,:] += ubar[1]*v
    for j = 1:n-1
        G, r = givens(R[j,j], R[j+1,j], j, j+1)
        R[j,j] = r
        R[j+1,j]= 0.
        R[:,j+1:n] = G*R[:,j+1:n]
        Q[:,:] = Q[:,:]*G'
    end
end

# ╔═╡ fba9aa90-a241-413e-a7f2-d330058b472f
let
	# Random matrix and u/v vectors to test things out
	A = rand(10,10)
	u = rand(10)
	v = rand(10)

	# Julia QR uses WY format for the Q factor
	# -- we need to change to a standard dense representation to update
	F = qr(A)
	Q = Matrix(F.Q)
	R = F.R

	# Apply the update and check that Q and R satisfy the desired relations
	qrupdate!(Q, R, u, v)

	md"""
A test case is always useful to avoid coding errors

- Check factorization OK: $(norm(Q*R-A-u*v'))
- Check orthonormality:   $(norm(Q'*Q-I))
- Check upper triangular: $(norm(tril(R,-1)))
"""
end

# ╔═╡ 820e020f-e4fc-4edb-8cfc-3cf53c44dea8
md"""
#### Questions

Complete the following code for Broyden with QR updating.
"""

# ╔═╡ 88103c3e-3eec-46c8-a26f-fcb9639e929c
function dense_broyden(x, f, J; rtol=1e-6, nsteps=100, monitor=(x, fx) -> nothing)

    x = copy(x)            # Copy x (so we can overwrite without clobbering the initial guess)
    J = Array{Float64}(J)  # Dense matrix representation of the initial Jacobian
    F = qr(J)              # Get a QR factorization
    Q = Matrix(F.Q)        # Convert the Q factor to standard dense form
    R = F.R                # Extract the R factor
    
    # Take an initial step to get things started
    fx = f(x)
    monitor(x, fx, J)
    s = -J\fx
    
    for step = 1:nsteps

        # Take Broyden step and update Jacobian (overwrite old arrays)
        x[:] += s
        fx[:] = f(x)
        
        # TODO: Replace these lines with QR-based version
        J[:,:] += fx*(s/norm(s)^2)'
        s[:] = -J\fx

        # Monitor progress and check for convergence
        monitor(x, fx)
        if norm(fx) < rtol
            return x, fx
        end

    end
    
    # We could error out here, but let's just return x and fx
    return x, fx
end

# ╔═╡ d52a2b96-82c2-4e7a-b166-43a4b03a0d35
let
	rhist = Array{Float64,1}([])
	J0 = Jautocatalytic(v0)
	v = naive_broyden(v0, autocatalytic, J0, rtol=1e-10, monitor=(x, fx)->push!(rhist, norm(fx)))
	plot(rhist, yscale=:log10, legend=false)
end

# ╔═╡ 344edaab-011f-4626-98bd-3600021263b0
md"""
### Large-scale Broyden

For large-scale problems, the $O(n^2)$ time and storage cost of a dense Broyden update may be prohibitive, and we would like an alternative with lower complexity.  As long as we have a fast solver for the initial Jacobian $J_0$, we can compute the first several Broyden steps using a bordered linear system

$$\begin{bmatrix} J_0 & -F_k \\ S_K^T & D_k \end{bmatrix}
  \begin{bmatrix} s^k \\ \mu \end{bmatrix} = 
  \begin{bmatrix} -f(x^k) \\ 0 \end{bmatrix},$$

where

$$\begin{align*}
  F_k &= \begin{bmatrix} f(x^1) & \ldots & f(x^k) \end{bmatrix}, \\
  S_k &= \begin{bmatrix} s^0 & \ldots s^{k-1} \end{bmatrix}, \\
  D_k &= \operatorname{diag}( \|s^0\|^2, \ldots, \|s^{k-1}\|^2)
\end{align*}$$

Defining vectors $g^k = -J_0^{-1} f(x^k)$ (chord steps), we can rewrite the system as

$$\begin{bmatrix} I & G_k \\ S_k^T & D_k \end{bmatrix} 
  \begin{bmatrix} s^k \\ \mu \end{bmatrix} =
  \begin{bmatrix} g^k \\ 0 \end{bmatrix}$$

Defining $w = -\mu$ and performing block Gaussian elimination then yields

$$\begin{align*}
(D_k - S_k^T G_k) w &= S^T g_k \\
s^k &= g^k + G_k w
\end{align*}$$

Hence, to go from step $k$ to step $k+1$ in this framework requires one new solve with $J_0$, $O(k^2)$ time to solve the Schur complement system using an existing factorization (or $O(k^3)$ to refactor each time), and $O(nk)$ time to form the new step and extend the Schur complement system for the next solve.  As long as $k$ is modest, this is likely an acceptable cost.  For larger $k$, though, it may become annoying.  Possible solutions include *limited memory Broyden*, which only takes into account the last $m$ steps of the iteration when computing the modified Jacobian; or *restarted Broyden*, which restarts from an approximate Jacobian of $J_0$ after every $m$ Broyden steps.

We illustrate the idea with a simple limited memory Broyden code.  We do not attempt anything clever with the Schur complement solve.
"""

# ╔═╡ b9f58009-73cb-429a-8e03-ce3f2f578a38
function limited_broyden(x, f, J0solve; rtol=1e-6, nsteps=100, m=10, 
                         monitor=(x, fx) -> nothing)

    x = copy(x)
    G = zeros(length(x), m)
    S = zeros(length(x), m)
    D = zeros(m, m)

    # Take an initial step to get things started
    fx = f(x)
    monitor(x, fx)
    s = -J0solve(fx)
    
    for step = 1:nsteps
        
        # Take Broyden step and evaluate function
        x[:] += s
        fx[:] = f(x)
        
        # Update G, S, and D
        k = mod(step-1, m)+1
        S[:,k] = s
        G[:,k] = -J0solve(fx)
        D[k,k] = norm(s)^2
        
        # Solve next step (keep track of at most m previous steps)
        p = min(step, m)
        w = (D[1:p,1:p] - S[:,1:p]'*G[:,1:p])\(S[:,1:p]'*G[:,k])
        s[:] = G[:,k] + G[:,1:p]*w

        # Monitor progress and check for convergence
        monitor(x, fx)
        if norm(fx) < rtol
            return x, fx
        end

    end
    
    # We could error out here, but let's just return x and fx
    return x, fx
end

# ╔═╡ 48bc71ee-07c7-4ea5-abbb-f9ff64b6a0f2
let
	rhist = Array{Float64,1}([])
	J0 = Jautocatalytic(v0)
	J0solve(x) = J0\x
	v = limited_broyden(v0, autocatalytic, J0solve, rtol=1e-10, monitor=(x, fx)->push!(rhist, norm(fx)))[1]
	plot(rhist, yscale=:log10, legend=false)
end

# ╔═╡ 06b51f55-0dfe-4993-932d-ac976968325e
md"""
#### Questions

Given that the cost of a tridiagonal solve is $O(n)$, what is the dominant cost per iteration in the limited-memory Broyden code above?  Could that cost be reduced by more clever book-keeping?
"""

# ╔═╡ 09e6cc30-d9b1-48da-b3ed-87dd0b808d02
md"""
## Reduced rank extrapolation

*Extrapolation methods* are sequence transformations that convert a slowly-converging sequence into one that is more rapidly convergent.  *Vector extrapolation* methods include reduced rank extrapolation, minimal polynomial extrapolation, and vector Padé methods.  We will focus on the example of reduced rank extrapolation (RRE), but all of these methods have a similar flavor.

Suppose $x_1, x_2, \ldots \in \mathbb{R}^n$ converges to some limit $x_*$, albeit slowly.  We also suppose an error model that describes $e_k = x_k-x_*$; in the case of RRE, the error model is

$$e_k \approx \sum_{j=1}^m u_j \alpha_j^k.$$

If this error model were exact (and if we knew the $\alpha_j$), we could define a degree $m+1$ polynomial

$$p(z) = \sum_{i=0}^m \gamma_i z^i$$

such that $p(z) = 1$ and $p(\alpha_j) = 0$ for each $j$.  Then

$$\begin{align*}
\sum_{i=0}^m \gamma_i x_{k+1}
&= \sum_{i=0}^m \gamma_i \left( x_* + \sum_{j=1}^m u_j \alpha_j^{k+1} \right) \\
&= p(1) x_* + \sum_{j=1}^m u_k \alpha_j^k p(\alpha_j) \\
&= x_*
\end{align*}$$

Of course, in practice, the error model is not exact, and we do not know the $\alpha_j$!  Nonetheless, we can come up with an appropriate choice of polynomials by asking for coefficients $\gamma$ such that

$$r_k = \sum_{i=0}^m \gamma_i x_{k+i+1} - \sum_{i=0}^m \gamma_i x_{k+1}$$

is as small as possible in a least squares sense.  Writing $f_{k+1} = x_{k+i+1}-x_{k+i}$ and $F_k = \begin{bmatrix} f_k & \ldots & f_{k+m} \end{bmatrix}$, we then have

$$\mbox{minimize } \frac{1}{2} \|F_k\gamma\|_2^2 \mbox{ s.t. } \sum_i \gamma_i = 1$$

Using the method of Lagrange multipliers to enforce the constraint, we have the Lagrangian

$$L(\gamma, \mu) = \frac{1}{2} \gamma^T (F_k^T F_k) \gamma + \mu (e^T \gamma - 1),$$

where $e$ is the vector of all ones; differentiating gives us the stationary equations

$$\begin{bmatrix} F_k^T F_k & e \\ e^T & 0 \end{bmatrix}
  \begin{bmatrix} \gamma \\ \mu \end{bmatrix} =
  \begin{bmatrix} 0 \\ 1 \end{bmatrix}.$$

If $F_k = Q_k R_k$ is an economy QR decomposition, the solution to this minimization problem is

$$\gamma = (R_k^{-1} y)/\|y\|^2, \quad y = R_k^{-T} e.$$

In principle, we can compute the QR decomposition of $F_{k+1}$ from the QR decomposition of $F_k$ relatively quickly (in $O(nm)$ time).  Hence, each time we want to compute one more vector in the extrapolated sequence, the cost is that of forming one more vector in the original sequence followed by $O(nm)$ additional work.  However, to demonstrate the idea, let's just write this in the most obvious way.
"""

# ╔═╡ dfb3a50d-362a-47d5-9857-c90db7fa8752
function rre(X)
    m = size(X)[2]-1
    if m == 0
        return X[:,1]
    end
    F = qr(X[:,2:end]-X[:,1:end-1])
    y = F.R'\ones(m)
    γ = (F.R\y)/norm(y)^2
    return X[:,2:end]*γ
end

# ╔═╡ 849ea4ac-2ed7-4c2b-905e-eeb692c52e2d
let
	rhist0 = []
	rhistr = []
	J0 = Jautocatalytic(0*v0)
	X = zeros(length(v0), 20)
	X[:,1] = 0*v0
	fx = autocatalytic(X[:,1])
	for k = 2:20
   		X[:,k] = X[:,k-1]-J0\fx
    	fx[:] = autocatalytic(X[:,k])
    	xrre = rre(X[:,1:k])
    	push!(rhist0, norm(fx[:]))
    	push!(rhistr, norm(autocatalytic(xrre)))
	end

	plot(rhist0, yscale=:log10, label="Original sequence")
	plot!(rhistr, label="RRE-accelerated")
end

# ╔═╡ 14e3ad55-f7d8-4cbf-b71a-3c7a3544a64b
md"""
When applied to a the iterates of a stationary iterative method for solving $Ax = b$,
reduced rank extrapolation is formally the same as the GMRES iteration. In numerical practice, though, the orthogonalization that takes place when forming the Krylov subspace bases in GMRES provides much better numerical stability than we would get from RRE. Whether we apply the method to linear or nonlinear problems, the matrices $F_k$ in RRE are often rather ill-conditioned, and the coefficient vectors $\gamma$ have large positive and negative entries, so that forming $\sum_i \gamma_i x_{k+i}$ may lead to significant cancellation.  For this reason, one may want to choose modest values of $m$ in practice.
"""

# ╔═╡ 8c802249-abee-4c19-80b3-18199695e904
md"""
#### Questions

Explain why the QR-based algorithm given above to minimize $\|F_k \gamma\|^2$ subject to $e^T \gamma = 1$ satisfies the stationary conditions.
"""

# ╔═╡ 80b61dd8-1928-4046-9da1-4a668e3a8b26
md"""
## Anderson acceleration

Anderson acceleration is an old method, common in computational chemistry, that has recently become popular for more varied problems thanks to the work of Tim Kelley, Homer Walker, and various others. In many ways, it is similar to reduced rank extrapolation applied to a sequence of iterates from a fixed point iteration. If we wish to find $x_*$ such that

$$f(x_*) = g(x_*)-x_* = 0$$

then reduced rank extrapolation -- without any linear algebra trickery -- at step $k$ looks like

$$\begin{align*}
m_k &= \min(m,k) \\
F_k &= \begin{bmatrix} f(x_{k-m_k}) & \ldots & f(x_k) \end{bmatrix} \\
\gamma^{(k)} &= \operatorname{argmin} \|F_k \gamma\|_2 \mbox{ s.t. } \sum_{i=0}^{m_k} = 1 \\
s_{k+1} &= \sum_{i=0}^{m_k} \gamma_i^{(k)} g(x_{k-m_k+i})
\end{align*}$$

In this application of reduced rank extrapolation, the output sequence  $s_k$ and the input sequence $x_k$ (defined by the iteration $x_{k+1} = g(x_k)$) are distinct entities. By contrast, in Anderson acceleration we have just one sequence:

$$\begin{align*}
m_k &= \min(m,k) \\
F_k &= \begin{bmatrix} f(x_{k-m_k}) & \ldots & f(x_k) \end{bmatrix} \\
\gamma^{(k)} &= \operatorname{argmin} \|F_k \gamma\|_2 \mbox{ s.t. } \sum_{i=0}^{m_k} = 1 \\
x_{k+1} &= \sum_{i=0}^{m_k} \gamma_i^{(k)} g(x_{k-m_k+i})
\end{align*}$$

The only difference in the pseudocode is that the last step generates a new $x_{k+1}$ (which feeds into the next iteration), rather than generating a separate output sequence $s_{k+1}$ and computing the next input iterate by a fixed point step. Thus, the difference between Anderson acceleration and reduced rank extrapolation is morally the same as the difference between Gauss-Seidel and Jacobi iteration: in the former case, we try to work with the most recent guesses available.

Unsurprisingly, Anderson acceleration (like RRE) is equivalent to GMRES when applied to linear fixed point iterations. Anderson acceleration can also be seen as a multi-secant generalization of Broyden’s iteration. For a good overview of the literature and of different ways of thinking about the method (as well as a variety of interesting applications), I recommend “[Anderson Acceleration for Fixed-Point Iterations](https://doi.org/10.1137/10078356X)” by Walker and Ni (SIAM J. Numer. Anal., Vol. 49, No. 4, pp. 1715–1735).
"""

# ╔═╡ db4a7981-e015-4318-9fd6-386fb3250372
function anderson_step(X, gX)
    m = size(gX)[2]
    F = qr(gX-X)
    y = F.R'\ones(m)
    γ = (F.R\y)/norm(y)^2
    return gX*γ
end

# ╔═╡ 83bdcfcc-7474-4c30-8a2a-c1ae47ccbd21
let
	rhistaa = []
	J0 = Jautocatalytic(0*v0)

	X  = zeros(length(v0), 20)
	gX = zeros(length(v0), 20)

	X[:,1] = 0*v0
	fx = autocatalytic(X[:,1])
	gX[:,1] = X[:,1] - J0\fx

	for k = 2:20
	    X[:,k] = anderson_step(X[:,1:k-1], gX[:,1:k-1])
	    fx[:] = autocatalytic(X[:,k])
	    gX[:,k] = X[:,k] - J0\fx
	    push!(rhistaa, norm(fx[:]))
	end

	plot(rhistaa, yscale=:log10, label="AA-accelerated")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[compat]
Plots = "~1.27.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "b8af57da2a9cf3236511afd7aca0bdfa3f229efc"

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
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

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
git-tree-sha1 = "af237c08bda486b74318c8070adb96efa6952530"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cd6efcf9dc746b06709df14e462f0a3fe0786b1e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.2+0"

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
git-tree-sha1 = "6f14549f7760d84b2db7a9b10b88cd3cc3025730"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.14"

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
git-tree-sha1 = "621f4f3b4977325b9128d5fae7a8b4829a0c2222"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.4"

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
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "88ee01b02fba3c771ac4dce0dfc4ecf0cb6fb772"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.5"

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
# ╟─c2fc87e6-b506-11ec-2093-c5e4c97aae19
# ╠═f44f2491-52ef-4b43-920a-666eac83368b
# ╠═88ccddfb-9645-4905-b6f9-8f7a60519e5f
# ╟─0627e5e4-452f-4778-948b-fdaa4c3f60ec
# ╠═704f1c07-b6b5-4c03-abbc-1951e37b6a7b
# ╠═372c511c-1a35-44fb-958c-ec51a0a034e6
# ╠═c62b0f6e-8709-4dc5-bc5f-3978b37f6b5c
# ╟─821e8d55-022e-44c9-b22d-a6eb42b48025
# ╠═7a932860-d0c6-4dfb-9438-1451b8f66cfb
# ╟─bc73550e-29ab-4696-b0a5-57bf2d195d2f
# ╠═d171b1ba-9bb1-4aee-8879-772383b8df3c
# ╟─184743b7-8c56-40f7-be96-ad35ff2bb46d
# ╟─4051701e-e695-4b2b-8a5a-80b076615e04
# ╠═40d528bc-596a-4e2d-ae0a-796bab25d680
# ╠═fba9aa90-a241-413e-a7f2-d330058b472f
# ╟─820e020f-e4fc-4edb-8cfc-3cf53c44dea8
# ╠═88103c3e-3eec-46c8-a26f-fcb9639e929c
# ╠═d52a2b96-82c2-4e7a-b166-43a4b03a0d35
# ╟─344edaab-011f-4626-98bd-3600021263b0
# ╠═b9f58009-73cb-429a-8e03-ce3f2f578a38
# ╠═48bc71ee-07c7-4ea5-abbb-f9ff64b6a0f2
# ╟─06b51f55-0dfe-4993-932d-ac976968325e
# ╟─09e6cc30-d9b1-48da-b3ed-87dd0b808d02
# ╠═dfb3a50d-362a-47d5-9857-c90db7fa8752
# ╠═849ea4ac-2ed7-4c2b-905e-eeb692c52e2d
# ╟─14e3ad55-f7d8-4cbf-b71a-3c7a3544a64b
# ╟─8c802249-abee-4c19-80b3-18199695e904
# ╟─80b61dd8-1928-4046-9da1-4a668e3a8b26
# ╠═db4a7981-e015-4318-9fd6-386fb3250372
# ╠═83bdcfcc-7474-4c30-8a2a-c1ae47ccbd21
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
