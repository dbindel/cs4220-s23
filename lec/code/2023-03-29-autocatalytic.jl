function autocatalytic(v)
  N = length(v)
  fv = exp.(v)
  fv -= 2*(N+1)^2*v
  fv[1:N-1] += (N+1)^2*v[2:N  ]
  fv[2:N  ] += (N+1)^2*v[1:N-1]
  fv
end
