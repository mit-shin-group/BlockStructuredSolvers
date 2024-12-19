using LinearAlgebra

import Pkg
include("TriDiagBlock.jl")
import .TriDiagBlock: TriDiagBlockData, ThomasSolve, ThomasFactorize

N = 95 # number of diagonal blocks
n = 8 # size of each block
P = 15 # number of separators
m  = trunc(Int, (N - P) / (P + 1))

#######################################
A_list = zeros(N, 8, 8)
for i = 1:N
    temp = rand(Float64, 8, 8)
    A_list[i, :, :] = temp * temp'

end

B_list = zeros(N-1, 8, 8)
for i = 1:N-1
    temp = rand(Float64, 8, 8)
    B_list[i, :, :] = temp * temp'
    # B_list[i, :, :] = Diagonal(ones(8)) * 0.01
end

xtrue_list = rand(N, n)
d_list = zeros(N, n)

d_list[1, :] = A_list[1, :, :] * xtrue_list[1, :] + B_list[1, :, :] * xtrue_list[2, :]

for i = 2:N-1

    d_list[i, :] = B_list[i-1, :, :]' * xtrue_list[i-1, :] + A_list[i, :, :] * xtrue_list[i, :] + B_list[i, :, :] * xtrue_list[i+1, :]

end

d_list[N, :] = B_list[N-1, :, :]' * xtrue_list[N-1, :] + A_list[N, :, :] * xtrue_list[N, :]
#################################################################

temp_list = zeros(N, n, n)

data = TriDiagBlockData(N, A_list, B_list, temp_list)

I_separator = []

for j = 1:P
    push!(I_separator, trunc(Int, (N+1)/(P+1) * j))
end

###########
MA = zeros((N-P) * n, (N-P) * n)

MA[1:n, 1:n] = A_list[1, :, :]
MA[1:n, 1+n:n+n] = B_list[1, :, :]
for i = 2:m-1
    MA[(i-1)*n+1:i*n, (i-2)*n+1:(i-1)*n] = B_list[i-1, :, :]'
    MA[(i-1)*n+1:i*n, (i-1)*n+1:i*n] = A_list[i, :, :]
    MA[(i-1)*n+1:i*n, (i)*n+1:(i+1)*n] = B_list[i, :, :]
end
MA[(m-1)*n+1:m*n, (m-2)*n+1:(m-1)*n] = B_list[m-1, :, :]'
MA[(m-1)*n+1:m*n, (m-1)*n+1:m*n] = A_list[m, :, :]

for j = 1:P
    MA[j*m*n+1:j*m*n+n, j*m*n+1:j*m*n+n] = A_list[I_separator[j]+1, :, :]
    MA[j*m*n+1:j*m*n+n, j*m*n+1+n:j*m*n+n+n] = B_list[I_separator[j]+1, :, :]
    for i = 2:m-1
        MA[j*m*n+(i-1)*n+1:j*m*n+i*n, j*m*n+(i-2)*n+1:j*m*n+(i-1)*n] = B_list[I_separator[j]+i-1, :, :]'
        MA[j*m*n+(i-1)*n+1:j*m*n+i*n, j*m*n+(i-1)*n+1:j*m*n+i*n] = A_list[I_separator[j]+i, :, :]
        MA[j*m*n+(i-1)*n+1:j*m*n+i*n, j*m*n+(i)*n+1:j*m*n+(i+1)*n] = B_list[I_separator[j]+i, :, :]
    end
    MA[j*m*n+(m-1)*n+1:j*m*n+m*n, j*m*n+(m-2)*n+1:j*m*n+(m-1)*n] = B_list[I_separator[j]+m-1, :, :]'
    MA[j*m*n+(m-1)*n+1:j*m*n+m*n, j*m*n+(m-1)*n+1:j*m*n+m*n] = A_list[I_separator[j]+m, :, :]
end


###################3
MB = zeros((N-P) * n, P * n)
MB[m*n-n+1:m*n, 1:n] = B_list[I_separator[1]-1, :, :]
for j = 2:P
    MB[(j-1)*m*n+1:(j-1)*m*n+n, (j-2)*n+1:(j-1)*n] = B_list[I_separator[j-1], :, :]'
    MB[j*m*n-n+1:j*m*n, (j-1)*n+1:(j)*n] = B_list[I_separator[j]-1, :, :]
end
MB[P*m*n+1:P*m*n+n, (P-1)*n+1:P*n] = B_list[I_separator[P], :, :]'

###################
MD = zeros(P * n, P * n)

for j = 1:P

    MD[(j-1)*n+1:j*n, (j-1)*n+1:j*n] = A_list[I_separator[j], :, :]

end

##################
v = zeros((N-P) * n)

for i = 1:m
    v[(i-1)*n+1:i*n] = d_list[i, :]
end

for j = 1:P

    for i = 1:m
        v[j*m*n+(i-1)*n+1:j*m*n+i*n] = d_list[I_separator[j]+i, :]
    end

end


####################################
u = zeros(P * n)

for j = 1:P

    u[(j-1)*n+1:j*n] = d_list[I_separator[j], :]

end


######
x_list = zeros(N, n)
x_separator_sol = inv(MD - MB' * inv(MA) * MB) * (u - MB' * inv(MA) * v)

x_list[I_separator, :] = reshape(x_separator_sol, n, P)'
######

d_list[I_separator[1]-1, :] -= B_list[I_separator[1]-1, :, :] * x_list[I_separator[1], :]
for j = 2:P
    d_list[I_separator[j-1]+1, :] -= B_list[I_separator[j-1], :, :]' * x_list[I_separator[j-1], :]
    d_list[I_separator[j]-1, :] -= B_list[I_separator[j]-1, :, :] * x_list[I_separator[j], :]
end
d_list[I_separator[P]+1, :] -= B_list[I_separator[P], :, :]' * x_list[I_separator[P], :]

##########################
x_temp_list = zeros(m, n)

for j = 1:P+1

    data = TriDiagBlockData(m, A_list[(j-1)*(m+1)+1:(j-1)*(m+1)+m, :, :], B_list[(j-1)*(m+1)+1:(j-1)*(m+1)+m-1, :, :],  zeros(m, n, n))

    ThomasFactorize(data)

    ThomasSolve(data, d_list[(j-1)*(m+1)+1:(j-1)*(m+1)+m, :], x_temp_list)
    x_list[(j-1)*(m+1)+1:(j-1)*(m+1)+m, :] = x_temp_list

end

x_list - xtrue_list

