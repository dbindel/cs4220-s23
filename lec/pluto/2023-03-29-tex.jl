include("2023-03-29.jl")

using PGFPlotsX
pgfplotsx()

savefig(test_toy_contraction(), "../fig/2023-03-29-toy-fp.tikz")
savefig(test_toy_newton(1, 2)[2], "../fig/2023-03-29-toy-newton.tikz")
savefig(test_autocatalytic(), "../fig/2023-03-29-test-autocatalytic.tikz")
savefig(test_autocatalytic_fp(), "../fig/2023-03-29-test-autocatalytic-fp.tikz")
