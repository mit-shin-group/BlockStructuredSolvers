using LinearAlgebra

import Pkg
include("TriDiagBlock.jl")
import .TriDiagBlock: TriDiagBlockData, ThomasSolve, ThomasFactorize

N = 97 # number of diagonal blocks
n = 8 # size of each block
P = 17 # number of separators
m  = trunc(Int, (N - P) / (P - 1))

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

I_separator = 1:(m+1):N

###########
MA = zeros((P-1) * m * n, (P-1) * m * n)

for j = 1:P-1
    MA[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*m*n+1:(j-1)*m*n+n] = A_list[I_separator[j]+1, :, :]
    MA[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*m*n+1+n:(j-1)*m*n+n+n] = B_list[I_separator[j]+1, :, :]
    for i = 2:m-1
        MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i-2)*n+1:(j-1)*m*n+(i-1)*n] = B_list[I_separator[j]+i-1, :, :]'
        MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i)*n+1:(j-1)*m*n+(i+1)*n] = B_list[I_separator[j]+i, :, :]
        MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n] = A_list[I_separator[j]+i, :, :]
    end
    MA[(j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n, (j-1)*m*n+(m-2)*n+1:(j-1)*m*n+(m-1)*n] = B_list[I_separator[j]+m-1, :, :]'
    MA[(j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n, (j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n] = A_list[I_separator[j]+m, :, :]
end


###################3
MB = zeros((P-1) * m * n, P * n)
for j = 1:P-1
    MB[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*n+1:(j-1)*n+n] = B_list[I_separator[j], :, :]'
    MB[j*m*n-n+1:j*m*n, (j-1)*n+n+1:(j-1)*n+n+n] = B_list[I_separator[j+1]-1, :, :]
end

###################
MD = zeros(P * n, P * n)

for j = 1:P

    MD[(j-1)*n+1:j*n, (j-1)*n+1:j*n] = A_list[I_separator[j], :, :]

end

##################
v = zeros((P-1) * m * n)

for j = 1:P-1

    for i = 1:m
        v[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n] = d_list[I_separator[j]+i, :]
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
for j = 1:P-1
    d_list[I_separator[j]+1, :] -= B_list[I_separator[j], :, :]' * x_list[I_separator[j], :]
    d_list[I_separator[j+1]-1, :] -= B_list[I_separator[j+1]-1, :, :] * x_list[I_separator[j+1], :]
end

##########################
# x_temp_list = zeros(m, n)

# for j = 1:P-1

#     data = TriDiagBlockData(m, A_list[I_separator[j]+1:I_separator[j+1]-1, :, :], B_list[I_separator[j]+1:I_separator[j+1]-1-1, :, :],  zeros(m, n, n))

#     ThomasFactorize(data)

#     ThomasSolve(data, d_list[I_separator[j]+1:I_separator[j+1]-1, :], x_temp_list)
#     x_list[I_separator[j]+1:I_separator[j+1]-1, :] = x_temp_list

# end

# x_list - xtrue_list

batch_A_list = zeros(P-1, m, n, n)
batch_B_list = zeros(P-1, m-1, n, n)
temp_B_list = zeros(P-1, m-1, n, n)

@views for j = 1:P-1

    batch_A_list[j, :, :, :] = A_list[I_separator[j]+1:I_separator[j+1]-1, :, :]
    batch_B_list[j, :, :, :] = B_list[I_separator[j]+1:I_separator[j+1]-1-1, :, :]

end

for j = 1:P-1

    temp_B_list[j, 1, :, :] = inv(batch_A_list[j, 1, :, :]) * batch_B_list[j, 1, :, :]

    for i = 2:m-1
        
        temp_B_list[j, i, :, :] = inv(batch_A_list[j, i, :, :] - batch_B_list[j, i-1, :, :]' * temp_B_list[j, i-1, :, :]) * batch_B_list[j, i, :, :]

    end
end


####################

batch_d_list = zeros(P-1, m, n)
temp_d_list = zeros(P-1, m, n)

@views for j = 1:P-1

    batch_d_list[j, :, :] = d_list[I_separator[j]+1:I_separator[j+1]-1, :]

end

for j = 1:P-1

    temp_d_list[j, 1, :] = inv(batch_A_list[j, 1, :, :]) * batch_d_list[j, 1, :]

    for i = 2:m
        
        temp_d_list[j, i, :] = inv(batch_A_list[j, i, :, :] - batch_B_list[j, i-1, :, :]' * temp_B_list[j, i-1, :, :]) * (batch_d_list[j, i, :] - batch_B_list[j, i-1, :, :]' * temp_d_list[j, i-1, :])

    end
end

for j = 1:P-1

    x_list[I_separator[j]+m, :] = temp_d_list[j, m, :]

    for i = m-1:-1:1

        x_list[I_separator[j]+i, :] = temp_d_list[j, i, :] - temp_B_list[j, i, :, :] *  x_list[I_separator[j]+i+1, :]

    end

end

x_list - xtrue_list