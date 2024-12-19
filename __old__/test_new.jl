import Pkg
include("pdp.jl")
import .pdp: LQProbData, PDP

N = 5
m = 2
n = 3

P = trunc(Int, (N + 1) / 2)

I_separator = [1]

for j = 1:P-1
    push!(I_separator, I_separator[end]+2)
end

push!(I_separator, N+1)

Q = rand(Float64, N+1, n, n)
R = rand(Float64, N+1, n, m)
S = rand(Float64, N+1, m, m)

A = rand(Float64, N+1, n, n)
B = rand(Float64, N+1, n, m)

z = rand(Float64, N+1, n)
r = rand(Float64, N+1, m)
s = rand(Float64, N+1, n)

s0 = zeros(n)

data = LQProbData(N, n, m, P, I_separator, Q, R, S, B, A, z, r, s, s0)

y = PDP(data)

display(y)