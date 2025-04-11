struct BlockTriDiagData{
    T, 
    MT <: AbstractMatrix{T},
    VMT :: Vector{MT}
    }

    N::Int
    m::Int
    n::Int
    P::Int

    I_separator::Vector{Int}

    LHS_vec::MT
    A_list::VMT
    B_list::VMT

    d::MT
    d_list::VMT

    LHS_A_list::VMT
    LHS_B_list::VMT

    factor_list::VMT
    temp_list::VMT
    temp_B_list::VMT

    M_2n_list::VMT
    factor_list_temp::VMT

    NextData::Union{BlockTriDiagData, Nothing}

end

function extract_AB_list!(LHS_vec::MT, A_list::Vector{MT}, B_list::Vector{MT}, N::Int, n::Int) where {T, MT<:AbstractMatrix{T}}

    LHS_vec_ptr = pointer(LHS_vec)

    for i in 1:N-1

        A_list[i] = unsafe_wrap(MT, LHS_vec_ptr + 2*n^2*(i-1)*sizeof(T), (n, n); own=false)
        B_list[i] = unsafe_wrap(MT, LHS_vec_ptr + 2*n^2*(i-1)*sizeof(T) + n^2*sizeof(T), (n, n); own=false)
    end

    A_list[N] = unsafe_wrap(MT, LHS_vec_ptr + 2*n^2*(N-1)*sizeof(T), (n, n); own=false)

end

# Extracts x_list_gpu from x_gpu using unsafe_wrap
function extract_d_list!(d::MT, d_list::Vector{MT}, N::Int, n::Int) where {T, MT<:AbstractMatrix{T}}

    # Get the raw pointer to the full vector data
    d_ptr = pointer(d)
    
    for i in 1:N
        d_list[i] = unsafe_wrap(MT, d_ptr + n *(i-1) * sizeof(T), (n, 1); 
                                own=false)
    end
 end

function initialize(N::Int, n::Int, T::Type, use_GPU::Bool)

    if use_GPU
        MT = CuMatrix{T}
    else
        MT = Matrix{T}
    end

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

        LHS_vec = MT(zeros((2*N-1)*n^2, 1))
        A_list = [MT(Matrix{T}(I, n, n)) for i in 1:N];
        B_list = [MT(zeros(n, n)) for i in 1:N-1];
        extract_AB_list!(LHS_vec, A_list, B_list, N, n)

        d = MT(zeros(N *n, 1))
        d_list = [MT(zeros(n, 1)) for i in 1:N];
        extract_d_list!(d, d_list, N, n)

        LHS_A_list = [MT(zeros(n, n)) for i in 1:P]
        LHS_B_list = [MT(zeros(n, n)) for i in 1:P-1]

        factor_list = [MT(zeros(n, 2*n)) for j = 1:N];
        temp_list = [MT(zeros(2*n, 1)) for i = 1:P-1];
        temp_B_list = [MT(zeros(n, n)) for i = 1:2*(P-1)];

        M_2n_list = [MT(zeros(2*n, 2*n)) for i in 1:P-1];
        factor_list_temp = [MT(zeros(n, 2*n)) for i in 1:N];

        data = BlockTriDiagData(
            N, 
            m, 
            n, 
            P, 
            I_separator,
            LHS_vec,
            A_list, 
            B_list,
            d,
            d_list,
            LHS_A_list,
            LHS_B_list,
            factor_list,
            temp_list,
            temp_B_list,
            M_2n_list,
            factor_list_temp,
            data
            );

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

function solve!(data::BlockTriDiagData, d_list::Vector{MT}) where {MT<:AbstractMatrix{T}}

    P = data.P
    n = data.n
    m = data.m
    I_separator = data.I_separator
    A_list = data.A_list
    B_list = data.B_list
    LHS_A_list = data.LHS_A_list
    LHS_B_list = data.LHS_B_list
    factor_list = data.factor_list
    temp_list = data.temp_list
    temp_B_list = data.temp_B_list

    # cache d_I_separator for frequent use
    d_I_separator = view(d_list, I_separator)

    # Compute RHS from Schur complement
    compute_schur_rhs!(factor_list, d_list, temp_list, I_separator, P, m, n)

    # Solve system
    if isnothing(data.NextData)
        cholesky_solve!(LHS_A_list, LHS_B_list, d_I_separator, P)
    else
        copy_vector_of_arrays!(data.NextData.d_list, d_I_separator)
        solve!(data.NextData, data.NextData.d_list)
        copy_vector_of_arrays!(d_I_separator, data.NextData.d_list)
    end

    # Update d after Schur solve
    update_boundary_solution!(temp_B_list, d_list, I_separator, P)

    # Solve for non-separators
    solve_non_separator_blocks!(A_list, B_list, d_list, I_separator, P, m)

    return nothing
end

function solve!(data::BlockTriDiagData, d<:AbstractMatrix{T})

    P = data.P
    n = data.n
    m = data.m
    I_separator = data.I_separator
    A_list = data.A_list
    B_list = data.B_list
    LHS_A_list = data.LHS_A_list
    LHS_B_list = data.LHS_B_list
    factor_list = data.factor_list
    temp_list = data.temp_list
    temp_B_list = data.temp_B_list

    copyto!(data.d, d)

    # cache d_I_separator for frequent use
    d_I_separator = view(d_list, I_separator)

    # Compute RHS from Schur complement
    compute_schur_rhs!(factor_list, d_list, temp_list, I_separator, P, m, n)

    # Solve system
    if isnothing(data.NextData)
        cholesky_solve!(LHS_A_list, LHS_B_list, d_I_separator, P)
    else
        copy_vector_of_arrays!(data.NextData.d_list, d_I_separator)
        solve!(data.NextData, data.NextData.d_list)
        copy_vector_of_arrays!(d_I_separator, data.NextData.d_list)
    end

    # Update d after Schur solve
    update_boundary_solution!(temp_B_list, d_list, I_separator, P)

    # Solve for non-separators
    solve_non_separator_blocks!(A_list, B_list, d_list, I_separator, P, m)

    copyto!(d, view(data.d, 1:length(d)))

end