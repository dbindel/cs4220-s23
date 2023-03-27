function Jautocatalytic(v)
  N = length(v)
  SymTridiagonal(exp.(v) .- 2*(N+1)^2, (N+1)^2 * ones(N-1))
end
