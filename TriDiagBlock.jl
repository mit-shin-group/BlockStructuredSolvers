module TriDiagBlock

using LinearAlgebra

export TriDiagBlockData, factorize, solve

mutable struct TriDiagBlockData{
    T, 
    MT <: Vector{Matrix{T}}, 
    MR <: Vector{Vector{Matrix{T}}}, 
    MS <: AbstractArray{T, 2}
    } # TODO preallocate all the struct

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

end

function factorize(
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

    batch_A_list = data.batch_A_list
    batch_B_list = data.batch_B_list
    temp_A_list = data.temp_A_list
    temp_B_list = data.temp_B_list

    ####################################################################
    @views for j = 1:P-1 #convert to while
        MA[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*m*n+1:(j-1)*m*n+n] = A_list[I_separator[j]+1]
        MA[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*m*n+1+n:(j-1)*m*n+n+n] = B_list[I_separator[j]+1]
        for i = 2:m-1
            MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i-2)*n+1:(j-1)*m*n+(i-1)*n] = B_list[I_separator[j]+i-1]'
            MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i)*n+1:(j-1)*m*n+(i+1)*n] = B_list[I_separator[j]+i]
            MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n] = A_list[I_separator[j]+i]
        end
        MA[(j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n, (j-1)*m*n+(m-2)*n+1:(j-1)*m*n+(m-1)*n] = B_list[I_separator[j]+m-1]'
        MA[(j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n, (j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n] = A_list[I_separator[j]+m]

        MB[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*n+1:(j-1)*n+n] = B_list[I_separator[j]]'
        MB[j*m*n-n+1:j*m*n, (j-1)*n+n+1:(j-1)*n+n+n] = B_list[I_separator[j+1]-1]
    end

    @views for j = 1:P

        MD[(j-1)*n+1:j*n, (j-1)*n+1:j*n] = A_list[I_separator[j]]

    end

    ####################################################################

    @views for j = 1:P-1

        LAPACK.gesv!(temp_A_list[j][1], temp_B_list[j][1])

        for i = 2:m-1

            temp_A_list[j][i] = batch_A_list[j][i] - batch_B_list[j][i-1]' * temp_B_list[j][i-1]
            LAPACK.gesv!(temp_A_list[j][i], temp_B_list[j][i])

        end
    end

end

function solve(
    data::TriDiagBlockData,
    d_list,
    u,
    v,
    batch_d_list,
    temp_d_list,
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
    temp_A_list = data.temp_A_list
    temp_B_list = data.temp_B_list

    for j = 1:P-1

        for i = 1:m
            v[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n] = d_list[I_separator[j]+i]
        end
    
    end
    
    for j = 1:P
    
        u[(j-1)*n+1:j*n] = d_list[I_separator[j]]
    
    end

    x_list[I_separator, :] = reshape(inv(MD - MB' * inv(MA) * MB) * (u - MB' * inv(MA) * v), n, P)'

    @views for j = 1:P-1
        d_list[I_separator[j]+1] -= B_list[I_separator[j]]' * x_list[I_separator[j], :]
        d_list[I_separator[j+1]-1] -= B_list[I_separator[j+1]-1] * x_list[I_separator[j+1], :]
    end

    @views for j = 1:P-1

        batch_d_list[j] = d_list[I_separator[j]+1:I_separator[j+1]-1]
    
    end
    
    for j = 1:P-1

        LAPACK.gesv!(batch_A_list[j][1], batch_d_list[j][1])
    
        for i = 2:m
    
            batch_d_list[j][i] -= batch_B_list[j][i-1]' * batch_d_list[j][i-1]
            temp_A_list[j][i] = batch_A_list[j][i] - batch_B_list[j][i-1]' * temp_B_list[j][i-1]
            LAPACK.gesv!(temp_A_list[j][i], batch_d_list[j][i])
    
        end
    end
    
    for j = 1:P-1

        x_list[I_separator[j]+m, :] = batch_d_list[j][m]
    
        for i = m-1:-1:1
    
            x_list[I_separator[j]+i, :] = batch_d_list[j][i] - temp_B_list[j][i] *  x_list[I_separator[j]+i+1, :]
    
        end
    
    end

    end

end