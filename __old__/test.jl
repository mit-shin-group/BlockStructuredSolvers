using LinearAlgebra

import Pkg
include("TriDiagBlock.jl")
import .TriDiagBlock: TriDiagBlockData, ThomasSolve, ThomasFactorize

N = 100
n = 8

A_list = zeros(N, 8, 8)

# Q = [[5.2478, -5.2896, 0] [-5.2896, 7.5183, -1.6938] [0, -1.6938, 1.3119]]
# R = zeros(Float64, 3, 2)
# S = [[4.0, 0.0] [0.0, 1.0]]
# A = [[-2, -.25, 0] [2, -2.1706, -1] [0, 1.8752, -1]] / N + Diagonal(zeros(3))
# B = [[-.873, 0, 0] [0, -.873, 0]] / N

for i = 1:N
    # A_list[i, :, :] = [Q R A'; R' S B'; A B zeros(3, 3)]
    temp = rand(Float64, 8, 8)
    A_list[i, :, :] = temp * temp'
    # A_list[i, :, :] = Diagonal(ones(8))

end

B_list = zeros(N-1, 8, 8)

# for i = 1:N-1
#     B_list[i, 6:8, 1:3] = - Diagonal(ones(3))
# end

for i = 1:N-1
    temp = rand(Float64, 8, 8)
    B_list[i, :, :] = temp * temp'
    # B_list[i, :, :] = Diagonal(ones(8)) * 0.01
end

xtrue_list = rand(N, n)
x_list = zeros(N, n)
d_list = zeros(N, n)

d_list[1, :] = A_list[1, :, :] * xtrue_list[1, :] + B_list[1, :, :] * xtrue_list[2, :]

for i = 2:N-1

    d_list[i, :] = B_list[i-1, :, :]' * xtrue_list[i-1, :] + A_list[i, :, :] * xtrue_list[i, :] + B_list[i, :, :] * xtrue_list[i+1, :]

end

d_list[N, :] = B_list[N-1, :, :]' * xtrue_list[N-1, :] + A_list[N, :, :] * xtrue_list[N, :]

Coeff = zeros(800, 800)

Coeff[1:8, 1:8] = A_list[1, :, :]
Coeff[1:8, 9:16] = B_list[1, :, :]
for j = 2:N-1
    Coeff[8*(j-1)+1:8*j, 8*j+1:8*(j+1)] = B_list[j, :, :]
    Coeff[8*(j-1)+1:8*j, 8*(j-2)+1:8*(j-1)] = B_list[j-1, :, :]'
    Coeff[8*(j-1)+1:8*j, 8*(j-1)+1:8*j] = A_list[j, :, :]
end
Coeff[end-7:end, end-7:end] = A_list[N, :, :]
Coeff[end-7:end, end-15:end-8] = B_list[N-1, :, :]'


RHS = zeros(800)
for j=1:N
    RHS[8*(j-1)+1:8*j] = d_list[j, :]
end

sol1 = Coeff \ RHS

temp_list = zeros(N, n, n)

data = TriDiagBlockData(N, A_list, B_list, temp_list)

ThomasFactorize(data)

ThomasSolve(data, d_list, x_list)


# sol_true = zeros(800)
# for i = 1:N
#     sol_true[(i-1) * 8 + 1:i*8] = xtrue_list[i, :]
# end

# ThomasFactorize(data)

# ThomasSolve(data, d_list, x_list)


```
  0.1237538154019741
  0.7326597523974353
  0.1101526663815353

  1.2734311706673076
  1.0871881484166641
  1.6329052564421978
```