module TriDiagBlock

using LinearAlgebra

export TriDiagBlockData, factorize, solve

struct TriDiagBlockData{
    T, 
    MT <: AbstractArray{T, 3},
    MR <: AbstractArray{T, 4}, 
    MS <: AbstractArray{T, 2},
    MU <: AbstractArray{T, 1}
    }

    N::Int
    m::Int
    n::Int
    P::Int

    I_separator::StepRange{Int64, Int64}

    A_list::MT
    B_list::MT

    batch_A_list::MR
    batch_B_list::MR
    temp_A_list::MR
    temp_B_list::MR

    MA::MS
    MB::MS
    MD::MS

    A::MS
    B::MS
    C::MS

    D1::MU
    D2::MU

    F::Cholesky{T, Matrix{T}}

end

function factorize!(
    data::TriDiagBlockData
)   

    N = data.N
    P = data.P
    n = data.n
    m = data.m

    I_separator = data.I_separator

    A_list = data.A_list
    B_list = data.B_list

    MA = data.MA
    MB = data.MB
    MD = data.MD

    # batch_A_list = data.batch_A_list
    batch_B_list = data.batch_B_list
    temp_A_list = data.temp_A_list
    temp_B_list = data.temp_B_list

    A = data.A
    B = data.B
    C = data.C

    F = data.F

    ####################################################################
    @views for j = 1:P-1 #TODO convert to while/improve indexing
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

    ####################################################################

    @views for j = 1:P-1

        copyto!(A, temp_A_list[j, 1, :, :])  # Avoids allocation, just copies data
        copyto!(B, temp_B_list[j, 1, :, :])  # Avoids allocation
        # LAPACK.potrf!('U', A)
        # LAPACK.potrs!('U', A, B)
        F = cholesky!(Hermitian(A))
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

end

function solve(
    data::TriDiagBlockData,
    d_list,
    u,
    v,
    batch_d_list,
    x_list
    )

    N = data.N
    P = data.P
    n = data.n
    m = data.m

    I_separator = data.I_separator

    MA = data.MA
    MB = data.MB
    MD = data.MD
    
    B_list = data.B_list
    batch_A_list = data.batch_A_list
    batch_B_list = data.batch_B_list
    temp_B_list = data.temp_B_list

    A = data.A
    B = data.B
    C = data.C
    D1 = data.D1
    D2 = data.D2

    F = data.F

    @views for j = 1:P-1

        for i = 1:m
            v[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n] = d_list[I_separator[j]+i, :]
        end
    
    end
    
    @views for j = 1:P

        u[(j-1)*n+1:j*n] = d_list[I_separator[j], :]
    
    end

    x_list[I_separator, :] = reshape(inv(MD - MB' * inv(MA) * MB) * (u - MB' * inv(MA) * v), n, P)' #TODO non allocation

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

    @views for j = 1:P-1
        
        copyto!(batch_d_list[j, :, :], d_list[I_separator[j]+1:I_separator[j+1]-1, :])
    
    end
    
    @views for j = 1:P-1

        copyto!(A, batch_A_list[j, 1, :, :])
        copyto!(D1, batch_d_list[j, 1, :])
        F = cholesky!(Hermitian(A))
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

    end

end