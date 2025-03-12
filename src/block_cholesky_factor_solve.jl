struct BlockTriDiagData{
    T, 
    MR <: Vector{Vector{AbstractMatrix{T}}},
    MT <: Vector{AbstractMatrix{T}},
    MS <: AbstractMatrix{T},
    MV <: Vector{AbstractVector{T}}
    }

    N::Int
    m::Int
    n::Int
    P::Int

    I_separator::StepRange{Int64, Int64}

    A_list::MT
    B_list::MT

    LHS_A_list::MT
    LHS_B_list::MT

    factor_list::MR

    RHS::MV

    MA_chol_A_list::MR
    MA_chol_B_list::MR
    LHS_chol_A_list::MT
    LHS_chol_B_list::MT

    M_2n::MS
    M_mn_2n_1::MT
    M_mn_2n_2::MT

    NextData::Union{BlockTriDiagData, Nothing}

    next_idx::Vector{Int}
    next_x::MV

end

function initialize(N, m, n, P, A_list_final, B_list_final, level)

    data = nothing;
    T = eltype(A_list_final[1]);

    for i = 1:level

        I_separator = 1:(m+1):N

        LHS_A_list = Vector{AbstractMatrix{T}}(undef, P);
        for i = 1:P
            LHS_A_list[i] = zeros(n, n);
        end

        LHS_B_list = Vector{AbstractMatrix{T}}(undef, P-1);
        for i = 1:P-1
            LHS_B_list[i] = zeros(n, n);
        end

        RHS = Vector{AbstractVector{T}}(undef, P);
        for i = 1:P
            RHS[i] = zeros(n);
        end

        MA_chol_A_list = Vector{Vector{AbstractMatrix{T}}}(undef, P-1);
        for i = 1:P-1
            MA_chol_A_list[i] = Vector{AbstractMatrix{T}}(undef, m);
            for j = 1:m
                MA_chol_A_list[i][j] = zeros(n, n);
            end
        end

        MA_chol_B_list = Vector{Vector{AbstractMatrix{T}}}(undef, P-1);
        for i = 1:P-1
            MA_chol_B_list[i] = Vector{AbstractMatrix{T}}(undef, m-1);
            for j = 1:m-1
                MA_chol_B_list[i][j] = zeros(n, n);
            end
        end
        LHS_chol_A_list = Vector{AbstractMatrix{T}}(undef, P);
        for i = 1:P
            LHS_chol_A_list[i] = zeros(n, n);
        end

        LHS_chol_B_list = Vector{AbstractMatrix{T}}(undef, P-1);
        for i = 1:P-1
            LHS_chol_B_list[i] = zeros(n, n);
        end
        factor_list = Vector{Vector{AbstractMatrix{T}}}(undef, P-1);
        for i = 1:P-1
            factor_list[i] = Vector{AbstractMatrix{T}}(undef, m);
            for j = 1:m
                factor_list[i][j] = zeros(n, 2*n);
            end
        end


        M_2n = zeros(2*n, 2*n);
        M_mn_2n_1 = Vector{AbstractMatrix{T}}(undef, m);
        for i = 1:m
            M_mn_2n_1[i] = zeros(n, 2*n);
        end
        M_mn_2n_2 = Vector{AbstractMatrix{T}}(undef, m);
        for i = 1:m
            M_mn_2n_2[i] = zeros(n, 2*n);
        end

        next_idx = Int[]

        for j = I_separator

            append!(next_idx, (j-1)*n+1:j*n)
            
        end

        next_x = Vector{AbstractVector{T}}(undef, P);
        for i = 1:P
            next_x[i] = zeros(n);
        end

        if i == level
            A_list = A_list_final
            B_list = B_list_final
        else
            A_list = Vector{AbstractMatrix{T}}(undef, N);
            for i = 1:N
                A_list[i] = zeros(n, n);
            end
            B_list = Vector{AbstractMatrix{T}}(undef, N-1);
            for i = 1:N-1
                B_list[i] = zeros(n, n);
            end
        end

        data = BlockTriDiagData(
            N, 
            m, 
            n, 
            P, 
            I_separator,
            A_list, 
            B_list,
            LHS_A_list,
            LHS_B_list,
            factor_list,
            RHS,
            MA_chol_A_list,
            MA_chol_B_list,
            LHS_chol_A_list,
            LHS_chol_B_list,
            M_2n,
            M_mn_2n_1,
            M_mn_2n_2,
            data,
            next_idx,
            next_x,
            );

        P = N;
        N = P * (m + 1) - m;

    end

    return data

end

function factorize!(data::BlockTriDiagData)

P = data.P
n = data.n
m = data.m

I_separator = data.I_separator

A_list = data.A_list
B_list = data.B_list

LHS_A_list = data.LHS_A_list
LHS_B_list = data.LHS_B_list

factor_list = data.factor_list

MA_chol_A_list = data.MA_chol_A_list
MA_chol_B_list = data.MA_chol_B_list

M_2n = data.M_2n
M_mn_2n_1 = data.M_mn_2n_1
M_mn_2n_2 = data.M_mn_2n_2

@inbounds for i = 1:P-1
    # Cache views for better performance
    M_2n .= zeros(2*n, 2*n);
    MA_chol_A_list[i] .= A_list[I_separator[i]+1:I_separator[i]+m]
    MA_chol_B_list[i] .= B_list[I_separator[i]+1:I_separator[i]+m-1]
    # Compute inverse of block tridiagonal matrices
    cholesky_factorize!(
        MA_chol_A_list[i],
        MA_chol_B_list[i],
        m
        )
        
    @views begin
        M_mn_2n_1[1][:, 1:n] .= B_list[I_separator[i]]'
        M_mn_2n_1[m][:, n+1:2*n] .= B_list[I_separator[i+1]-1]
    end

    M_mn_2n_2 .= M_mn_2n_1

    cholesky_solve!(MA_chol_A_list[i], MA_chol_B_list[i], M_mn_2n_2, m, n)

    factor_list[i] = M_mn_2n_2

    for j = 1:m
        mygemm!('T', 'N', 1.0, M_mn_2n_1[j], M_mn_2n_2[j], 1.0, M_2n)
    end
    
    @views begin
        LHS_A_list[i] .-= M_2n[1:n, 1:n]
        LHS_A_list[i+1] .-= M_2n[n+1:2*n, n+1:2*n]
        LHS_B_list[i] .-= M_2n[1:n, n+1:2*n]
    end

    LHS_A_list[i] .+= A_list[I_separator[i]]
end

# copyto!(A, view(LHS_A_list, :, :, P))
LHS_A_list[P] .+= A_list[I_separator[P]]
# copyto!(view(LHS_A_list, :, :, P), A)

if isnothing(data.NextData)
    LHS_chol_A_list = data.LHS_chol_A_list
    LHS_chol_B_list = data.LHS_chol_B_list

    LHS_chol_A_list .= LHS_A_list
    LHS_chol_B_list .= LHS_B_list

    cholesky_factorize!(LHS_chol_A_list, LHS_chol_B_list, P)
else
    data.NextData.A_list .= LHS_A_list
    data.NextData.B_list .= LHS_B_list
    factorize!(data.NextData)
end

end

function solve!(data::BlockTriDiagData, d_list, x)
    P = data.P
    n = data.n
    m = data.m

    I_separator = data.I_separator
    B_list = data.B_list
    MA_chol_A_list = data.MA_chol_A_list
    MA_chol_B_list = data.MA_chol_B_list
    factor_list = data.factor_list
    RHS = data.RHS

    # Assign RHS from d
    # @inbounds @simd for j = 1:P
    #     copyto!(RHS[j], d_list[I_separator[j]])
    # end
    RHS .= d_list[I_separator]

    # Compute RHS from Schur complement
    @inbounds for i = 1:P-1 #TODO
        temp = zeros(2*n);
        for j = 1:m
            mygemv!('T', -1.0, factor_list[i][j], d_list[I_separator[i]+j], 1.0, temp)
        end
        RHS[i] .+= temp[1:n]
        RHS[i+1] .+= temp[n+1:2*n]
    end

    # Solve system
    if isnothing(data.NextData)
        LHS_chol_A_list = data.LHS_chol_A_list
        LHS_chol_B_list = data.LHS_chol_B_list

        cholesky_solve!(LHS_chol_A_list, LHS_chol_B_list, RHS, P, n)

        # Assign RHS to x for separators
        # @inbounds @simd for i = 1:P
        #     copyto!(view(x, I_separator[i]*n-n+1:I_separator[i]*n), view(RHS, (i-1)*n+1:i*n))
        # end
        x[I_separator] .= RHS
    else
        data.next_x .= x[I_separator]
        solve!(data.NextData, RHS, data.next_x)
        x[I_separator] .= data.next_x
    end

    # Update d after Schur solve
    @inbounds for j = 1:P-1

        mygemv!('T', -1.0, B_list[I_separator[j]], x[I_separator[j]], 1.0, d_list[I_separator[j]+1])

        mygemv!('N', -1.0, B_list[I_separator[j+1]-1], x[I_separator[j+1]], 1.0, d_list[I_separator[j+1]-1])
    end

    # Solve for non-separators
    @inbounds for i = 1:P-1
        cholesky_solve!(MA_chol_A_list[i], MA_chol_B_list[i], d_list[I_separator[i]+1:I_separator[i+1]-1], m, n)
        x[I_separator[i]+1:I_separator[i+1]-1] .= d_list[I_separator[i]+1:I_separator[i+1]-1]
    end

    return nothing
end