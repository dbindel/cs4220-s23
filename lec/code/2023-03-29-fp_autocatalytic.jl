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
