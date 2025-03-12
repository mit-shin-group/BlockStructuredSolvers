struct BlockTriDiagData{
    T, 
    MS <: AbstractMatrix{T},
    MR <: Vector{<:Vector{<:MS}},
    MT <: Vector{<:MS}
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

    RHS::MT

    MA_chol_A_list::MR
    MA_chol_B_list::MR
    LHS_chol_A_list::MT
    LHS_chol_B_list::MT

    M_2n::MS
    M_mn_2n_list_1::MT
    M_mn_2n_list_2::MT
    v_2n::MS

    NextData::Union{BlockTriDiagData, Nothing}

end

function initialize(N, m, n, P, A_list_final, B_list_final, level)

    data = nothing;
    T = eltype(A_list_final[1]);

    for i = 1:level

        I_separator = 1:(m+1):N

        LHS_A_list = Vector{Matrix{T}}(undef, P); #TODO use similar to infer from input
        for i = 1:P
            LHS_A_list[i] = zeros(n, n);
        end

        LHS_B_list = Vector{Matrix{T}}(undef, P-1);
        for i = 1:P-1
            LHS_B_list[i] = zeros(n, n);
        end

        RHS = Vector{Matrix{T}}(undef, P);
        for i = 1:P
            RHS[i] = zeros(n, 1);
        end

        MA_chol_A_list = Vector{Vector{Matrix{T}}}(undef, P-1);
        for i = 1:P-1
            MA_chol_A_list[i] = Vector{Matrix{T}}(undef, m);
            for j = 1:m
                MA_chol_A_list[i][j] = zeros(n, n);
            end
        end

        MA_chol_B_list = Vector{Vector{Matrix{T}}}(undef, P-1);
        for i = 1:P-1
            MA_chol_B_list[i] = Vector{Matrix{T}}(undef, m-1);
            for j = 1:m-1
                MA_chol_B_list[i][j] = zeros(n, n);
            end
        end
        LHS_chol_A_list = Vector{Matrix{T}}(undef, P);
        for i = 1:P
            LHS_chol_A_list[i] = zeros(n, n);
        end

        LHS_chol_B_list = Vector{Matrix{T}}(undef, P-1);
        for i = 1:P-1
            LHS_chol_B_list[i] = zeros(n, n);
        end
        factor_list = Vector{Vector{Matrix{T}}}(undef, P-1);
        for i = 1:P-1
            factor_list[i] = Vector{Matrix{T}}(undef, m);
            for j = 1:m
                factor_list[i][j] = zeros(n, 2*n);
            end
        end

        M_2n = zeros(2*n, 2*n);
        M_mn_2n_list_1 = Vector{Matrix{T}}(undef, m);
        for i = 1:m
            M_mn_2n_list_1[i] = zeros(n, 2*n);
        end
        M_mn_2n_list_2 = Vector{Matrix{T}}(undef, m);
        for i = 1:m
            M_mn_2n_list_2[i] = zeros(n, 2*n);
        end
        v_2n = zeros(2*n, 1);

        if i == level
            A_list = A_list_final
            B_list = B_list_final
        else
            A_list = Vector{Matrix{T}}(undef, N);
            for i = 1:N
                A_list[i] = zeros(n, n);
            end
            B_list = Vector{Matrix{T}}(undef, N-1);
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
            M_mn_2n_list_1,
            M_mn_2n_list_2,
            v_2n,
            data
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

    # Cache all arrays at the start
    A_list = data.A_list
    B_list = data.B_list
    LHS_A_list = data.LHS_A_list
    LHS_B_list = data.LHS_B_list
    factor_list = data.factor_list
    MA_chol_A_list = data.MA_chol_A_list
    MA_chol_B_list = data.MA_chol_B_list
    M_2n = data.M_2n
    M_mn_2n_list_1 = data.M_mn_2n_list_1
    M_mn_2n_list_2 = data.M_mn_2n_list_2

    # Main factorization loop
    @inbounds for i = 1:P-1
        # Reset M_2n to zero
        fill!(M_2n, 0.0)

        # Copy data for factorization
        copy_vector_of_arrays!(MA_chol_A_list[i], view(A_list, I_separator[i]+1:I_separator[i]+m))
        copy_vector_of_arrays!(MA_chol_B_list[i], view(B_list, I_separator[i]+1:I_separator[i]+m-1))

        # Perform Cholesky factorization
        cholesky_factorize!(MA_chol_A_list[i], MA_chol_B_list[i], m) #TODO tiny allocation here about 2, belongs to potrf!

        # Set up M_mn_2n_list_! for Schur complement
        view(M_mn_2n_list_1[1], :, 1:n) .= B_list[I_separator[i]]'
        view(M_mn_2n_list_1[m], :, n+1:2*n) .= B_list[I_separator[i+1]-1]

        # Copy M_mn_2n_list_1 to M_mn_2n_list_2
        copy_vector_of_arrays!(M_mn_2n_list_2, M_mn_2n_list_1)

        # Solve using Cholesky factors
        cholesky_solve!(MA_chol_A_list[i], MA_chol_B_list[i], M_mn_2n_list_2, m, n)

        # Copy results to factor_list
        copy_vector_of_arrays!(factor_list[i], M_mn_2n_list_2)

        # Compute Schur complement
        for j = 1:m
            mygemm!('T', 'N', 1.0, M_mn_2n_list_1[j], M_mn_2n_list_2[j], 1.0, M_2n)
        end

        # Update LHS matrices
        LHS_A_list[i] .-= view(M_2n, 1:n, 1:n)
        LHS_A_list[i+1] .-= view(M_2n, n+1:2*n, n+1:2*n)
        LHS_B_list[i] .-= view(M_2n, 1:n, n+1:2*n)

        LHS_A_list[i] .+= A_list[I_separator[i]]
    end

    # Final update for LHS_A_list
    LHS_A_list[P] .+= A_list[I_separator[P]]

    # Recursive factorization
    if isnothing(data.NextData)
        LHS_chol_A_list = data.LHS_chol_A_list
        LHS_chol_B_list = data.LHS_chol_B_list

        copy_vector_of_arrays!(LHS_chol_A_list, LHS_A_list)
        copy_vector_of_arrays!(LHS_chol_B_list, LHS_B_list)

        cholesky_factorize!(LHS_chol_A_list, LHS_chol_B_list, P)
    else
        copy_vector_of_arrays!(data.NextData.A_list, LHS_A_list)
        copy_vector_of_arrays!(data.NextData.B_list, LHS_B_list)
        factorize!(data.NextData)
    end
end

function solve!(data::BlockTriDiagData, d_list, x)
    P = data.P
    n = data.n
    m = data.m
    I_separator = data.I_separator

    # Cache frequently accessed arrays
    B_list = data.B_list
    MA_chol_A_list = data.MA_chol_A_list
    MA_chol_B_list = data.MA_chol_B_list
    factor_list = data.factor_list
    RHS = data.RHS
    temp = data.v_2n

    # Copy d_list to RHS
    copy_vector_of_arrays!(RHS, view(d_list, I_separator))

    # Compute RHS from Schur complement
    for i = 1:P-1
        fill!(temp, 0.0)
        for j = 1:m
            mygemm!('T', 'N', -1.0, factor_list[i][j], d_list[I_separator[i]+j], 1.0, temp)
        end
        RHS[i] .+= view(temp, 1:n, :)
        RHS[i+1] .+= view(temp, n+1:2*n, :)
    end

    # Solve system
    if isnothing(data.NextData)
        LHS_chol_A_list = data.LHS_chol_A_list
        LHS_chol_B_list = data.LHS_chol_B_list

        cholesky_solve!(LHS_chol_A_list, LHS_chol_B_list, RHS, P, n)

        # Assign RHS to x for separators
        copy_vector_of_arrays!(view(x, I_separator), RHS)
    else
        solve!(data.NextData, RHS, view(x, I_separator))
    end

    # Update d after Schur solve
    @inbounds for j = 1:P-1
        mygemm!('T', 'N', -1.0, B_list[I_separator[j]], x[I_separator[j]], 1.0, d_list[I_separator[j]+1])
        mygemm!('N', 'N', -1.0, B_list[I_separator[j+1]-1], x[I_separator[j+1]], 1.0, d_list[I_separator[j+1]-1])
    end

    # Solve for non-separators
    @inbounds for i = 1:P-1
        cholesky_solve!(MA_chol_A_list[i], MA_chol_B_list[i], view(d_list, I_separator[i]+1:I_separator[i+1]-1), m, n)
        copy_vector_of_arrays!(view(x, I_separator[i]+1:I_separator[i+1]-1), view(d_list, I_separator[i]+1:I_separator[i+1]-1))
    end

    return nothing
end