import Pkg
include("pdp.jl")
import .pdp: LQProbData, LQProbSol, PDP, solveWholeSystem, RI, factorize, RI2, recover

using LinearAlgebra

N = 100
m = 2
n = 3

Q_i = [[5.2478, -5.2896, 0] [-5.2896, 7.5183, -1.6938] [0, -1.6938, 1.3119]]
Q_i = rand(n, n)
Q_i = Q_i' * Q_i
Q = repeat(Q_i, 1, 1, N+1)
Q = permutedims(Q, (3, 1, 2))

R = zeros(Float64, N, n, m)

S_i = [[4.0, 0.0] [0.0, 1.0]]
S = repeat(S_i, 1, 1, N)
S = permutedims(S, (3, 1, 2))

A_i = [[-2, -.25, 0] [2, -2.1706, -1] [0, 1.8752, -1]] / N + Diagonal(zeros(n))
A = repeat(A_i, 1, 1, N)
A = permutedims(A, (3, 1, 2))

B_i = [[-.873, 0, 0] [0, -.873, 0]] / N
B = repeat(B_i, 1, 1, N)
B = permutedims(B, (3, 1, 2))

z = rand(Float64, N+1, n)
r = rand(Float64, N, m)
s = rand(Float64, N, n)

s0 = zeros(n)

############################
P = trunc(Int, (N + 1) / 2)

I_separator = [1]

for j = 1:P-1
    push!(I_separator, I_separator[end]+2)
end

push!(I_separator, N+1)

q = zeros(N+1, n)
y = zeros(N+1, n)
v = zeros(N+1, m)

sol = LQProbSol(q, y, v)
data = PDP(N, n, m, P, I_separator, Q, R, S, B, A, z, r, s, s0)

```
factorize(data)
RI2(data, sol)
recover(data, sol)
```

```
y1 = solveWholeSystem(data)
y3 = RI(data)
```

