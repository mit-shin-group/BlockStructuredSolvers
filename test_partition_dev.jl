using LinearAlgebra

import Pkg
include("TriDiagBlock.jl")
import .TriDiagBlock: TriDiagBlockData, factorize, solve

N = 97 # number of diagonal blocks
n = 8 # size of each block
P = 17 # number of separators
m  = trunc(Int, (N - P) / (P - 1))

#######################################
A_list = zeros(N, n, n);
for i = 1:N
    temp = randn(Float64, n, n)
    A_list[i, :, :] = temp * temp' + n * I
end

B_list = zeros(N-1, n, n);
for i = 1:N-1
    temp = randn(Float64, n, n)
    B_list[i, :, :] = temp
end

x_true = rand(N, n);
d_list = zeros(N, n);
d_list[1, :] = A_list[1, :, :] * x_true[1, :] + B_list[1, :, :] * x_true[2, :];

for i = 2:N-1

    d_list[i, :] = B_list[i-1, :, :]' * x_true[i-1, :] + A_list[i, :, :] * x_true[i, :] + B_list[i, :, :] * x_true[i+1, :];

end

d_list[N, :] = B_list[N-1, :, :]' * x_true[N-1, :] + A_list[N, :, :] * x_true[N, :];
#################################################################

I_separator = 1:(m+1):N

####################################################################
MA = zeros((P-1) * m * n, (P-1) * m * n);
MB = zeros((P-1) * m * n, P * n);
MD = zeros(P * n, P * n);

@views for j = 1:P-1 #convert to while
    MA[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*m*n+1:(j-1)*m*n+n] = A_list[I_separator[j]+1, :, :]
    MA[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*m*n+1+n:(j-1)*m*n+n+n] = B_list[I_separator[j]+1, :, :]
    for i = 2:m-1
        MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i-2)*n+1:(j-1)*m*n+(i-1)*n] = B_list[I_separator[j]+i-1, :, :]'
        MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i)*n+1:(j-1)*m*n+(i+1)*n] = B_list[I_separator[j]+i, :, :]
        MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n] = A_list[I_separator[j]+i, :, :]
    end
    MA[(j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n, (j-1)*m*n+(m-2)*n+1:(j-1)*m*n+(m-1)*n] = B_list[I_separator[j]+m-1, :, :]'
    MA[(j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n, (j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n] = A_list[I_separator[j]+m, :, :]

    MB[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*n+1:(j-1)*n+n] = B_list[I_separator[j], :, :]'
    MB[j*m*n-n+1:j*m*n, (j-1)*n+n+1:(j-1)*n+n+n] = B_list[I_separator[j+1]-1, :, :]
end

@views for j = 1:P

    MD[(j-1)*n+1:j*n, (j-1)*n+1:j*n] = A_list[I_separator[j], :, :]

end

##################
v = zeros((P-1) * m * n)

@views for j = 1:P-1

    for i = 1:m
        v[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n] = d_list[I_separator[j]+i, :]
    end

end
####################################
u = zeros(P * n)

@views for j = 1:P

    u[(j-1)*n+1:j*n] = d_list[I_separator[j], :]

end


######
x_list = zeros(N, n);
x_separator_sol = inv(MD - MB' * inv(MA) * MB) * (u - MB' * inv(MA) * v);

x_list[I_separator, :] = reshape(x_separator_sol, n, P)';


######
@views for j = 1:P-1
    copyto!(D1, d_list[I_separator[j]+1, :])
    copyto!(B, B_list[I_separator[j], :, :]')
    copyto!(D2, x_list[I_separator[j], :])
    mul!(D1, B, D2, -1.0, 1.0)
    copyto!(d_list[I_separator[j]+1, :], D1)

    copyto!(D1, d_list[I_separator[j+1]-1, :])
    copyto!(B, B_list[I_separator[j+1]-1, :, :])
    copyto!(D2, x_list[I_separator[j+1], :])
    mul!(D1, B, D2, -1.0, 1.0)
    copyto!(d_list[I_separator[j+1]-1, :], D1)
end

batch_A_list = zeros(P-1, m, n, n);
batch_B_list = zeros(P-1, m-1, n, n);

@views for j = 1:P-1

    batch_A_list[j, :, :, :] = A_list[I_separator[j]+1:I_separator[j+1]-1, :, :]
    batch_B_list[j, :, :, :] = B_list[I_separator[j]+1:I_separator[j+1]-1-1, :, :]

end

temp_A_list = deepcopy(batch_A_list);
temp_B_list = deepcopy(batch_B_list);

A = similar(temp_A_list, n, n)
B = similar(temp_B_list, n, n)
C = similar(temp_A_list, n, n)
D1 = similar(temp_A_list, n)
D2 = similar(temp_A_list, n)
F = cholesky(Matrix{Float64}(I, n, n))

@views for j = 1:P-1

    copyto!(A, temp_A_list[j, 1, :, :])  # Avoids allocation, just copies data
    copyto!(B, temp_B_list[j, 1, :, :])  # Avoids allocation
    # LAPACK.potrf!('U', A)
    # LAPACK.potrs!('U', A, B)
    F = cholesky!(A)
    ldiv!(F, B)
    copyto!(temp_B_list[j, 1, :, :], B)  # Store the result back if needed

    for i = 2:m-1

        copyto!(A, batch_B_list[j, i-1, :, :]')
        copyto!(B, temp_B_list[j, i-1, :, :])
        copyto!(C, temp_A_list[j, i, :, :])

        mul!(C, A, B, -1.0, 1.0)

        copyto!(B, temp_B_list[j, i, :, :])  # Ensure B is contiguous
        # LAPACK.potrf!('U', C)
        # LAPACK.potrs!('U', C, B)  
        F = cholesky!(Hermitian(C))
        ldiv!(F, B)
        copyto!(temp_B_list[j, i, :, :], B)


    end
end


####################

batch_d_list = zeros(P-1, m, n)

@views for j = 1:P-1

    batch_d_list[j, :, :] = d_list[I_separator[j]+1:I_separator[j+1]-1, :]

end

# temp_d_list = copy(batch_d_list);

@views for j = 1:P-1

    copyto!(A, batch_A_list[j, 1, :, :])
    copyto!(D1, batch_d_list[j, 1, :])
    F = cholesky!(A)
    ldiv!(F, D1)
    copyto!(batch_d_list[j, 1, :], D1)

    for i = 2:m

        copyto!(D2, batch_d_list[j, i, :])
        copyto!(B, batch_B_list[j, i-1, :, :]')
        copyto!(C, temp_B_list[j, i-1, :, :])
        mul!(D2, B, D1, -1.0, 1.0)
        copyto!(A, batch_A_list[j, i, :, :])
        mul!(A, B, C, -1.0, 1.0)
        # LAPACK.potrf!('U', A)
        # LAPACK.potrs!('U', A, D2)
        F = cholesky!(Hermitian(A))
        ldiv!(F, D2)
        copyto!(batch_d_list[j, i, :], D2)
        copyto!(D1, batch_d_list[j, i, :])

    end
end

@views for j = 1:P-1

    copyto!(x_list[I_separator[j]+m, :], batch_d_list[j, m, :])

    for i = m-1:-1:1
        
        copyto!(D1, batch_d_list[j, i, :])
        copyto!(B, temp_B_list[j, i, :, :])
        copyto!(D2, x_list[I_separator[j]+i+1, :]) #TODO optimize the reuse D1/D2
        mul!(D1, B, D2, -1.0, 1.0)
        copyto!(x_list[I_separator[j]+i, :], D1)
        # x_list[I_separator[j]+i, :] = batch_d_list[j, i, :] - temp_B_list[j, i, :, :] *  x_list[I_separator[j]+i+1, :]

    end

end

x_list - x_true