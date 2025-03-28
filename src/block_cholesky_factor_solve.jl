struct BlockTriDiagData{
    T, 
    MT <: Vector{<:AbstractMatrix{T}}
    }

    N::Int
    m::Int
    n::Int
    P::Int

    I_separator::Vector{Int}
    I_non_separator::Vector{Int}

    A_list::MT
    B_list::MT

    LHS_A_list::MT
    LHS_B_list::MT

    factor_list::MT
    temp_list::MT
    temp_B_list::MT

    RHS::MT

    M_2n_list::MT
    factor_list_temp::MT

    NextData::Union{BlockTriDiagData, Nothing}

end

function initialize(N, n, A_list_final, B_list_final, parallel)

    data = nothing;
    level = 0;
    N_list = [];
    P_list = [];
    m_list = [];
    I_separator_list = [];
    
    while N > 10

        m = 4
        I_separator = 1:(m+1):N        
        if I_separator[end] != N
            I_separator = [I_separator; N]
        else    
            I_separator = [I_separator[1:end-1]; N]
        end
        
        P = length(I_separator)
        pushfirst!(N_list, N);
        pushfirst!(P_list, P);
        pushfirst!(m_list, m);
        pushfirst!(I_separator_list, I_separator);
        println("N = $N, P = $P, m = $m")
        N = P
        level += 1;

    end
    
    for i = 1:level

        P = P_list[i]
        N = N_list[i]
        m = m_list[i]
       
        I_separator = I_separator_list[i]
        I_non_separator = setdiff(1:N, I_separator)

        # TODO slow?
        if i == 1
            LHS_A_list = [zero(similar(A_list_final[1], n, n)) for i in 1:P];
            LHS_B_list = [zero(similar(B_list_final[1], n, n)) for i in 1:P-1];
        else
            LHS_A_list = deepcopy(data.A_list)
            LHS_B_list = deepcopy(data.B_list)
        end

        RHS = [zero(similar(A_list_final[1], n, 1)) for i in 1:P];

        factor_list = [similar(A_list_final[1], n, 2*n) for j = 1:N];
        temp_list = [zero(similar(A_list_final[1], 2*n, 1)) for i = 1:P-1];
        temp_B_list = [similar(A_list_final[1], n, n) for i = 1:2*(P-1)];

        M_2n_list = [zero(similar(A_list_final[1], 2*n, 2*n)) for i in 1:P-1];
        factor_list_temp = [zero(similar(A_list_final[1], n, 2*n)) for i in 1:N];

        if i == level
            A_list = A_list_final
            B_list = B_list_final
        else
            A_list = [zero(similar(A_list_final[1], n, n)) for i in 1:N];
            B_list = [zero(similar(B_list_final[1], n, n)) for i in 1:N-1];
        end

        data = BlockTriDiagData(
            N, 
            m, 
            n, 
            P, 
            I_separator,
            I_non_separator,
            A_list, 
            B_list,
            LHS_A_list,
            LHS_B_list,
            factor_list,
            temp_list,
            temp_B_list,
            RHS,
            M_2n_list,
            factor_list_temp,
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
    A_list = data.A_list
    B_list = data.B_list
    LHS_A_list = data.LHS_A_list
    LHS_B_list = data.LHS_B_list
    factor_list = data.factor_list
    M_2n_list = data.M_2n_list
    factor_list_temp = data.factor_list_temp
    temp_B_list = data.temp_B_list

    # Copy data for factorization
    copy_vector_of_arrays!(view(temp_B_list, 1:2:2*(P-1)), view(B_list, I_separator[1:P-1]))
    copy_vector_of_arrays!(view(temp_B_list, 2:2:2*(P-1)), view(B_list, I_separator[2:P].-1))
    add_vector_of_arrays!(LHS_A_list, view(A_list, I_separator))

    # Main factorization loop
    compute_schur_complement!(A_list, B_list, LHS_A_list, LHS_B_list, temp_B_list, factor_list, factor_list_temp, M_2n_list, I_separator, P, m, n)

    # Recursive factorization
    if isnothing(data.NextData)
        cholesky_factorize!(LHS_A_list, LHS_B_list, P)
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
    I_non_separator = data.I_non_separator
    RHS = data.RHS

    # Copy d_list to RHS
    copy_vector_of_arrays!(RHS, view(d_list, I_separator))

    # Compute RHS from Schur complement
    compute_schur_rhs!(data.factor_list, d_list, data.temp_list, RHS, I_separator, P, m, n)

    # Solve system
    if isnothing(data.NextData)
        cholesky_solve!(data.LHS_A_list, data.LHS_B_list, RHS, P)
        copy_vector_of_arrays!(view(x, I_separator), RHS)
    else
        solve!(data.NextData, RHS, view(x, I_separator))
    end

    # Update d after Schur solve
    update_boundary_solution!(data.temp_B_list, x, d_list, I_separator, P)

    # Solve for non-separators
    solve_non_separator_blocks!(data.A_list, data.B_list, d_list, I_separator, P, m)

    copy_vector_of_arrays!(view(x, I_non_separator), view(d_list, I_non_separator))

    return nothing
end