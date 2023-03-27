function newton_autocatalytic(α, N=100, nsteps=50, rtol=1e-8; 
                              monitor=(v, resid)->nothing)
  v_all = [α*x*(1-x) for x in range(0.0, 1.0, length=N+2)]
  v = v_all[2:N+1]
  for step = 1:nsteps
    fv = autocatalytic(v)
    resid = norm(fv)
    monitor(v, resid)
    if resid < rtol
      v_all[2:N+1] = v
      return v_all
    end
    v -= Jautocatalytic(v)\fv
  end
  error("Newton did not converge after $nsteps steps")
end
