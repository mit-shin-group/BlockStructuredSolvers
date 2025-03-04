mutable struct BlockTriDiagData{ #TODO create initialize function
    T, 
    MR <: AbstractArray{T, 4},
    MT <: AbstractArray{T, 3},
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

    LHS_A_list::MT
    LHS_B_list::MT

    factor_list::MT

    RHS::MU

    MA_chol_A_list::MR
    MA_chol_B_list::MR
    LHS_chol_A_list::MT
    LHS_chol_B_list::MT

    M_n_1::MS
    M_n_2::MS
    M_2n::MS
    M_n_2n_1::MS
    M_n_2n_2::MS
    M_mn_2n_1::MS
    M_mn_2n_2::MS

    v_n_1::MU
    v_n_2::MU

    NextData::Union{BlockTriDiagData, Nothing}

    next_idx::Vector{Int}
    next_x::MU

end

function initialize(N, m, n, P, A_list, B_list, level)

    I_separator = 1:(m+1):N

    LHS_A_list = zeros(P, n, n);
    LHS_B_list = zeros(P-1, n, n);

    MA_list = zeros(P-1, m*n, m*n);

    RHS = zeros(P * n);

    MA_chol_A_list = zeros(P-1, m, n, n);
    MA_chol_B_list = zeros(P-1, m-1, n, n);

    LHS_chol_A_list = zeros(P, n, n);
    LHS_chol_B_list = zeros(P-1, n, n);

    factor_list = zeros(P-1, m*n, 2*n);

    M_n_1 = similar(A_list, n, n);
    M_n_2 = similar(A_list, n, n);

    M_2n = similar(A_list, 2*n, 2*n);
    M_n_2n_1 = zeros(n, 2*n);
    M_n_2n_2 = zeros(n, 2*n);
    M_mn_2n_1 = zeros(m*n, 2*n);
    M_mn_2n_2 = zeros(m*n, 2*n);

    v_n_1 = zeros(n);
    v_n_2 = zeros(n);

    next_idx = Int[]

    for j = I_separator

        append!(next_idx, (j-1)*n+1:j*n)
        
    end

    next_x = zeros(P*n);

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
        M_n_1,
        M_n_2,
        M_2n,
        M_n_2n_1,
        M_n_2n_2,
        M_mn_2n_1,
        M_mn_2n_2,
        v_n_1,
        v_n_2,
        nothing,
        next_idx,
        next_x
    );

    prev_data = data;

    for i = 2:level

        N = P;
        m = 2;
        P = Int((N + m) / (m+1));

        I_separator = 1:(m+1):N

        LHS_A_list = zeros(P, n, n);
        LHS_B_list = zeros(P-1, n, n);

        RHS = zeros(P * n);

        MA_chol_A_list = zeros(P-1, m, n, n);
        MA_chol_B_list = zeros(P-1, m-1, n, n);
    
        LHS_chol_A_list = zeros(P, n, n);
        LHS_chol_B_list = zeros(P-1, n, n);

        factor_list = zeros(P-1, m*n, 2*n);

        M_n_1 = similar(A_list, n, n);
        M_n_2 = similar(A_list, n, n);

        M_2n = similar(A_list, 2*n, 2*n);
        M_n_2n_1 = zeros(n, 2*n);
        M_n_2n_2 = zeros(n, 2*n);
        M_mn_2n_1 = zeros(m*n, 2*n);
        M_mn_2n_2 = zeros(m*n, 2*n);

        v_n_1 = zeros(n);
        v_n_2 = zeros(n);

        next_idx = Int[]

        for j = I_separator

            append!(next_idx, (j-1)*n+1:j*n)
            
        end

        next_x = zeros(P*n);

        next_data = BlockTriDiagData(
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
            M_n_1,
            M_n_2,
            M_2n,
            M_n_2n_1,
            M_n_2n_2,
            M_mn_2n_1,
            M_mn_2n_2,
            v_n_1,
            v_n_2,
            nothing,
            next_idx,
            next_x,
            );

        prev_data.NextData = next_data;
        prev_data = next_data;
    end

    return data

end

function factorize!(
    data::BlockTriDiagData
)

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

A = data.M_n_1
B = data.M_n_2
M_2n = data.M_2n
M_mn_2n_1 = data.M_mn_2n_1
M_mn_2n_2 = data.M_mn_2n_2

u = data.M_n_2n_1
v = data.M_n_2n_2

@inbounds for i = 1:P-1
    # Cache views for better performance
    A_block = view(A_list, I_separator[i]+1:I_separator[i]+m, :, :)
    B_block = view(B_list, I_separator[i]+1:I_separator[i]+m-1, :, :)
    MA_chol_A = view(MA_chol_A_list, i, :, :, :)
    MA_chol_B = view(MA_chol_B_list, i, :, :, :)

    # Compute inverse of block tridiagonal matrices
    cholesky_factorize!(
        A_block,
        B_block,
        MA_chol_A,
        MA_chol_B,
        A,
        B,
        m, 
        n)
    
    # Cache frequently accessed views
    B_view1 = view(B_list, I_separator[i], :, :)
    B_view2 = view(B_list, I_separator[i+1]-1, :, :)
    
    @views begin
        M_mn_2n_1[1:n, 1:n] .= B_view1'
        M_mn_2n_1[m*n-n+1:m*n, n+1:2*n] .= B_view2
    end

    M_mn_2n_2 .= M_mn_2n_1

    cholesky_solve!(MA_chol_A, MA_chol_B, M_mn_2n_2, A, u, v, m, n)

    factor_list[i, :, :] = M_mn_2n_2

    mul!(M_2n, M_mn_2n_1', M_mn_2n_2)

    # Cache views for LHS updates
    lhs_a1 = view(LHS_A_list, i, :, :)
    lhs_a2 = view(LHS_A_list, i+1, :, :)
    lhs_b = view(LHS_B_list, i, :, :)
    
    @views begin
        lhs_a1 .-= M_2n[1:n, 1:n]
        lhs_a2 .-= M_2n[n+1:2*n, n+1:2*n]
        lhs_b .-= M_2n[1:n, n+1:2*n]
    end

    lhs_a1 .+= view(A_list, I_separator[i], :, :)
end

copyto!(A, view(LHS_A_list, P, :, :))
A .+= view(A_list, I_separator[P], :, :)
copyto!(view(LHS_A_list, P, :, :), A)

if isnothing(data.NextData)
    LHS_chol_A_list = data.LHS_chol_A_list
    LHS_chol_B_list = data.LHS_chol_B_list

    cholesky_factorize!(LHS_A_list, LHS_B_list, LHS_chol_A_list, LHS_chol_B_list, A, B, P, n)
else
    data.NextData.A_list = LHS_A_list
    data.NextData.B_list = LHS_B_list
    factorize!(data.NextData)
end

end

function solve!(data::BlockTriDiagData, d, x)
    P = data.P
    n = data.n
    m = data.m

    I_separator = data.I_separator
    B_list = data.B_list
    MA_chol_A_list = data.MA_chol_A_list
    MA_chol_B_list = data.MA_chol_B_list
    factor_list = data.factor_list

    B = data.M_n_2
    M_mn_2n_1 = data.M_mn_2n_1

    RHS = data.RHS
    u = data.v_n_1
    v = data.v_n_2

    # Assign RHS from d
    @inbounds @simd for j = 1:P
        copyto!(view(RHS, (j-1)*n+1:j*n), view(d, I_separator[j]*n-n+1:I_separator[j]*n))
    end

    # Compute RHS from Schur complement
    @inbounds for i = 1:P-1
        M_mn_2n_1 .= view(factor_list, i, :, :)
        # Cache views to avoid repeated view creation
        rhs_view = view(RHS, (i-1)*n+1:(i+1)*n)
        d_view = view(d, I_separator[i]*n+1:I_separator[i+1]*n-n)
        mul!(rhs_view, M_mn_2n_1', d_view, -1.0, 1.0)
    end

    # Solve system
    if isnothing(data.NextData)
        LHS_chol_A_list = data.LHS_chol_A_list
        LHS_chol_B_list = data.LHS_chol_B_list

        cholesky_solve!(LHS_chol_A_list, LHS_chol_B_list, RHS, B, u, v, P, n)

        # Assign RHS to x for separators
        @inbounds @simd for i = 1:P
            copyto!(view(x, I_separator[i]*n-n+1:I_separator[i]*n), view(RHS, (i-1)*n+1:i*n))
        end
    else
        data.next_x .= view(x, data.next_idx)
        solve!(data.NextData, RHS, data.next_x)
        view(x, data.next_idx) .= data.next_x
    end

    # Update d after Schur solve
    @inbounds for j = 1:P-1
        # Cache views and matrices
        B .= view(B_list, I_separator[j], :, :)
        x_view = view(x, I_separator[j]*n-n+1:I_separator[j]*n)
        d_view = view(d, I_separator[j]*n+1:I_separator[j]*n+n)
        gemm!('T', 'N', -1.0, B, x_view, 1.0, d_view)

        B .= view(B_list, I_separator[j+1]-1, :, :)
        x_view = view(x, I_separator[j+1]*n-n+1:I_separator[j+1]*n)
        d_view = view(d, I_separator[j+1]*n-n-n+1:I_separator[j+1]*n-n)
        mul!(d_view, B, x_view, -1.0, 1.0)
    end

    # Solve for non-separators
    @inbounds for i = 1:P-1
        d_view = view(d, I_separator[i]*n+1:I_separator[i+1]*n-n)
        cholesky_solve!(view(MA_chol_A_list, i, :, :, :), view(MA_chol_B_list, i, :, :, :), d_view, B, u, v, m, n)
        copyto!(view(x, I_separator[i]*n+1:I_separator[i+1]*n-n), d_view)
    end

    return nothing
end