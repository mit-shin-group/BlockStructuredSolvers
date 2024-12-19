module pdp

using LinearAlgebra

export LQProbData, solve, PDP, solveWholeSystem, RI, factorize, RI2, recover

mutable struct LQProbData{T, MT <: AbstractArray{T,3}, MS <: AbstractArray{T,2}} # TODO preallocate all the struct

    N::Int
    n::Int
    m::Int
    P::Int
    I_separator::Array{Int64, 1}

    Q::MT
    R::MT
    S::MT
    
    B::MT
    A::MT

    z::MS
    r::MS
    s::MS

    s0::Array{T, 1}

    Gamma::MT  # = zeros(N+1, m, n)
    gamma::MS # = zeros(N+1, m)
    Theta::MT # = zeros(N+1, n, n)
    beta::MS # = zeros(N+1, n)
    D::MT # = zeros(N+1, n, n)
    F::MT # = zeros(N+1, m, n)

    Qt::MT # = zeros(P, n, n)
    Rt::MT # = zeros(P, n, m)
    St::MT # = zeros(P, m, m)
    Jt::MT # = zeros(P, n, n)
    At::MT # = zeros(P, n, n)
    Bt::MT # = zeros(P, n, m)

    st::MS # = zeros(P, n)
    tt::MS # = zeros(P, n)
    rt::MS # = zeros(P, m)

    Ah::MT # = zeros(P, n, n)
    Jh::MT # = zeros(P, n, n)
    Qh::MT # = zeros(P, n, n)
    sh::MS # = zeros(P, n)
    th::MS # = zeros(P, n)

end

mutable struct LQProbSol{T, MS <: AbstractArray{T,2}} # TODO preallocate all the struct

    y::MS
    q::MS
    v::MS

end

function factorize(data::LQProbData)

    N = data.N
    n = data.n
    m = data.m
    P = data.P
    I_separator = data.I_separator
    
    Q = data.Q
    R = data.R
    S = data.S
    
    A = data.A
    B = data.B
    s = data.s
    r = data.r
    z = data.z
    s0 = data.s0
    
    # TODO add assert
    # N = size(data.Q, 1)
    # P = trunc(Int, (N + 1) / 2)
    # I_separator = [1]
    
    # TODO maynot need to save in struct
    Gamma = data.Gamma
    gamma = data.gamma
    Theta = data.Theta
    beta = data.beta
    D = data.D
    F = data.F

    Qt = data.Qt
    Rt = data.Rt
    St = data.St
    Jt = data.Jt

    At = data.At
    Bt = data.Bt

    st = data.st
    tt = data.tt
    rt = data.rt
    
    for j = 1:P
        
        I_jp1 = I_separator[j+1]
        I_j = I_separator[j]
    
        # Theta[I_jp1, :, :] = zeros(n, n)
        # beta[I_jp1, : ,:] = zeros(2, 1)
        D[I_jp1, :, :] = Diagonal(ones(n))
    
        for i = (I_jp1 - 1):-1:(I_j+1)
    
            Q_i = @views Q[i, :, :]
            R_i = @views R[i, :, :]
            S_i = @views S[i, :, :]
    
            A_i = @views A[i, :, :]
            B_i = @views B[i, :, :]
            A_i_T = transpose(A_i)
            B_i_T = transpose(B_i)
            
            z_i = @views z[i, :]
            r_i = @views r[i, :]
            s_i = @views s[i, :]
    
            Theta_ip1 = @views Theta[i+1, :, :]
            beta_ip1 = @views beta[i+1, :]
            D_ip1 = @views D[i+1, :, :]

            temp_block = inv(S_i + B_i_T * Theta_ip1 * B_i)
            
            ### TODO
            Gamma_i = - temp_block * (R_i' + B_i_T * Theta_ip1 * A_i) #TODO check inversion preallocation
            gamma_i = - temp_block * (r_i + B_i_T * (Theta_ip1 * s_i + beta_ip1))
            F_i = - temp_block * B_i_T * D_ip1
            D_i = (A_i + B_i * Gamma_i)' * D_ip1
            Theta_i = (Q_i + A_i_T * Theta_ip1 * A_i) + (R_i + A_i_T * Theta_ip1 * B_i) * Gamma_i
            beta_i = A_i_T * Theta_ip1 * (B_i * gamma_i + s_i) + A_i_T * beta_ip1 + z_i + R_i * gamma_i
    
            Gamma[i, :, :] = Gamma_i
            gamma[i, :] = gamma_i
            F[i, :, :] = F_i
            D[i, :, :] = D_i
            Theta[i, :, :] = Theta_i
            beta[i, :] = beta_i
        end
    
    end

    @views for j = 1:P

        I_jp1 = I_separator[j+1]
        I_j = I_separator[j]

        At[j, :, :] = D[I_j + 1, :, :]' * A[I_j, :, :]
        Bt[j, :, :] = D[I_j + 1, :, :]' * B[I_j, :, :]

        J_temp = zeros(n, n)
        s_temp = D[I_j + 1, :, :]' * s[I_j,:, :]
        for i = (I_j + 1):(I_jp1-1)
            J_temp += D[i+1, :, :]' * B[i, :, :] * F[i, :, :]
            s_temp += D[i+1, :, :]' * (B[i, :, :] * gamma[i, :] + s[i, :]) 
        end
        Jt[j, :, :] = J_temp
        st[j, :, :] = s_temp

        tt[j, :] = z[I_j, :] + transpose(A[I_j, :, :]) * (Theta[I_j + 1, :, :] * s[I_j, :] + beta[I_j + 1, :])
        Rt[j, :, :] = R[I_j, :, :] + transpose(A[I_j, :, :]) * Theta[I_j + 1, :, :] * B[I_j, :, :] 
        Qt[j, :, :] = Q[I_j, :, :] + transpose(A[I_j, :, :]) * Theta[I_j + 1, :, :] * A[I_j, :, :] 

        rt[j, :] = r[I_j, :] + transpose(B[I_j, :, :]) * (Theta[I_j + 1, :, :] * s[I_j, :] + beta[I_j + 1, :])
        St[j, :, :] = S[I_j, :, :] + transpose(B[I_j, :, :]) * Theta[I_j + 1, :, :] * B[I_j, :, :] 
        
    end

end

function solve(data::LQProbData)

    N = data.N
    n = data.n
    m = data.m
    P = data.P
    I_separator = data.I_separator

    Q = data.Q # put these three in data3
    z = data.z
    s0 = data.s0

    A = data.A
    B = data.B
    s = data.s

    Gamma = data.Gamma
    gamma = data.gamma
    Theta = data.Theta
    beta = data.beta
    D = data.D
    F = data.F

    Qt = data.Qt
    Rt = data.Rt
    St = data.St
    Jt = data.Jt

    At = data.At
    Bt = data.Bt

    st = data.st
    tt = data.tt
    rt = data.rt

    blk_size = n + m + n

    Coeff = zeros(blk_size * P + 2 * n, blk_size * P + 2 * n)

    @views for j = 1:P
        Coeff[blk_size*(j-1)+1:blk_size*(j-1)+n, blk_size*(j-1)+n+1:blk_size*(j-1)+2*n] = - Diagonal(ones(n))
        Coeff[blk_size*(j-1)+n+1:blk_size*(j-1)+2*n, blk_size*(j-1)+1:blk_size*(j-1)+n] = - Diagonal(ones(n))
        Coeff[blk_size*(j-1)+n+1:blk_size*j+n, blk_size*(j-1)+n+1:blk_size*j+n] = [Qt[j, :, :] Rt[j, :, :] At[j, :, :]'; Rt[j, :, :]' St[j, :, :] Bt[j, :, :]'; At[j, :, :] Bt[j, :, :] Jt[j, :, :]]
    end

    Coeff[end-2*n+1:end-n, end-n+1:end] = - Diagonal(ones(n))
    Coeff[end-n+1:end, end-2*n+1:end-n] = - Diagonal(ones(n))
    Coeff[end-n+1:end, end-n+1:end] = Q[end, :, :]

    RHS = -s0
    for j=1:P
        RHS = vcat(RHS, -tt[j, :], -rt[j, :], -st[j, :])
    end

    RHS = vcat(RHS, -z[end, :])
    sol = Coeff \ RHS

    q = data.q
    y = data.y
    v = data.v

    q[N+1, :] = sol[end-2*n+1:end-n]
    y[N+1, :] = sol[end-n+1:end]

    blk_size = n + m + n

    for j = P:-1:1

        I_jp1 = I_separator[j+1]
        I_j = I_separator[j]

        q[I_j, :] = sol[(j-1)*blk_size+1:(j-1)*blk_size+n]
        y[I_j, :] = sol[(j-1)*blk_size+n+1:(j-1)*blk_size+2*n]
        v[I_j, :] = sol[(j-1)*blk_size+2*n+1:j*blk_size]

    end

    return y

end

function RI(data::LQProbData)

    N = data.N
    n = data.n
    m = data.m

    Q = data.Q
    R = data.R
    S = data.S

    A = data.A
    B = data.B
    s = data.s
    r = data.r
    z = data.z

    Ah = zeros(N, n, n)
    Jh = zeros(N, n, n)
    Qh = zeros(N, n, n)
    sh = zeros(N, n)
    th = zeros(N, n)

    for i = 1:N
        S_inv = inv(S[i, :, :])

        Ah[i, :, :] = A[i, :, :] - B[i, :, :] * S_inv * R[i, :, :]'
        Jh[i, :, :] = - B[i, :, :] * S_inv * B[i, :, :]'
        Qh[i, :, :] = Q[i, :, :] - R[i, :, :] * S_inv * R[i, :, :]'

        th[i, :] = z[i, :] - R[i, :, :] * S_inv * r[i, :]
        sh[i, :] = s[i, :] - B[i, :, :] * S_inv * r[i, :]

    end

    K = zeros(N+1, n, n)
    b = zeros(N+1, n)
    temp_blocks = zeros(N, n, n)

    K[N+1, :, :] = Q[N+1, :, :]
    b[N+1, :] = z[N+1, :]
    temp_blocks[1, :, :] = inv(Diagonal(ones(n)) - K[2, :, :] * Jh[1, :, :])

    for i = N:-1:2

        temp_block_1 = inv(Diagonal(ones(n)) - K[i+1, :, :] * Jh[i, :, :])

        K[i, :, :] = Ah[i, :, :]' * temp_block_1 * K[i+1, :, :] * Ah[i, :, :] + Qh[i, :, :]
        b[i, :] = Ah[i, :, :]' * temp_block_1 * (K[i+1, :, :] * sh[i, :] + b[i+1, :]) + th[i, :]

        temp_blocks[i, :, :] = temp_block_1
        
    end

    y = zeros(N+1, n)
    
    for i = 1:N

        temp_block_1 = temp_blocks[i, :, :]
        y[i+1, :] = temp_block_1' * (Ah[i, :, :] * y[i, :] + sh[i, :] + Jh[i, :, :] * b[i+1, :])

    end

    return y

end

function solveWholeSystem(data::LQProbData)

    N = data.N
    n = data.n
    m = data.m
    
    Q = data.Q
    R = data.R
    S = data.S
    
    A = data.A
    B = data.B
    s = data.s
    r = data.r
    z = data.z
    s0 = data.s0

    blk_size = n + m + n

    Coeff = zeros(blk_size * N + 2 * n, blk_size * N + 2 * n)

    for j = 1:N
        Coeff[blk_size*(j-1)+1:blk_size*(j-1)+n, blk_size*(j-1)+n+1:blk_size*(j-1)+2*n] = - Diagonal(ones(n))
        Coeff[blk_size*(j-1)+n+1:blk_size*(j-1)+2*n, blk_size*(j-1)+1:blk_size*(j-1)+n] = - Diagonal(ones(n))
        Coeff[blk_size*(j-1)+n+1:blk_size*j+n, blk_size*(j-1)+n+1:blk_size*j+n] = [Q[j, :, :] R[j, :, :] A[j, :, :]'; R[j, :, :]' S[j, :, :] B[j, :, :]'; A[j, :, :] B[j, :, :] zeros(n, n)]
    end

    Coeff[end-2*n+1:end-n, end-n+1:end] = - Diagonal(ones(n))
    Coeff[end-n+1:end, end-2*n+1:end-n] = - Diagonal(ones(n))
    Coeff[end-n+1:end, end-n+1:end] = Q[end, :, :]

    RHS = -s0
    for j=1:N
        RHS = vcat(RHS, -z[j, :], -r[j, :], -s[j, :])
    end

    RHS = vcat(RHS, -z[end, :])
    sol1 = Coeff \ RHS
    display(sol1)

    y = zeros(N+1, n)

    for i = 1:N+1
        y[i, :] = sol1[n+1+(i-1)*blk_size:2*n+(i-1)*blk_size]
    end

    return y
end

function RI2(data::LQProbData, sol::LQProbSol) # separate RI factorize

    N = data.N
    n = data.n
    m = data.m
    P = data.P
    I_separator = data.I_separator

    Q_last = data.Q[N+1, :, :]
    z_last = data.z[N+1, :]
    Q = data.Qt
    R = data.Rt
    S = data.St
    J = data.Jt

    A = data.At
    B = data.Bt
    s = data.st
    r = data.rt
    z = data.tt
    s0 = data.s0

    Ah = data.Ah
    Jh = data.Jh
    Qh = data.Qh
    sh = data.sh 
    th = data.th

    y = sol.y
    q = sol.q
    v = sol.v

    for i = 1:P
        S_inv = inv(S[i, :, :])

        Ah[i, :, :] = A[i, :, :] - B[i, :, :] * S_inv * R[i, :, :]'
        Jh[i, :, :] = J[i, :, :] - B[i, :, :] * S_inv * B[i, :, :]'
        Qh[i, :, :] = Q[i, :, :] - R[i, :, :] * S_inv * R[i, :, :]'

        th[i, :] = z[i, :] - R[i, :, :] * S_inv * r[i, :]
        sh[i, :] = s[i, :] - B[i, :, :] * S_inv * r[i, :]

    end

    K = zeros(P+1, n, n)
    b = zeros(P+1, n)
    temp_blocks = zeros(P, n, n)

    K[P+1, :, :] = Q_last
    b[P+1, :] = z_last
    temp_blocks[1, :, :] = inv(Diagonal(ones(n)) - K[2, :, :] * Jh[1, :, :])

    for i = P:-1:2

        temp_block_1 = inv(Diagonal(ones(n)) - K[i+1, :, :] * Jh[i, :, :])

        K[i, :, :] = Ah[i, :, :]' * temp_block_1 * K[i+1, :, :] * Ah[i, :, :] + Qh[i, :, :]
        b[i, :] = Ah[i, :, :]' * temp_block_1 * (K[i+1, :, :] * sh[i, :] + b[i+1, :, :]) + th[i, :]

        temp_blocks[i, :, :] = temp_block_1
        
    end
    
    for j = 1:P

        I_jp1 = I_separator[j+1]
        I_j = I_separator[j]

        temp_block_1 = temp_blocks[j, :, :]
        y[I_jp1, :] = temp_block_1' * (Ah[j, :, :] * y[I_j, :] + sh[j, :] + Jh[j, :, :] * b[j+1, :])
        q[I_jp1, :] = K[j, :, :] * y[I_jp1, :] + b[j, :]
        v[I_jp1, :] = - inv(S[j, :, :]) * (r[j, :] + R[j, :, :]' * y[I_j, :] + B[j, :, :]' * q[I_jp1, :])
        qt_jp1 = q[I_jp1, :]

    end

end

function recover(data::LQProbData, sol::LQProbSol)

    P = data.P
    I_separator = data.I_separator

    A = data.A
    B = data.B
    s = data.s

    q = sol.q
    y = sol.y
    v = sol.v

    for j = P:-1:1

        I_jp1 = I_separator[j+1]
        I_j = I_separator[j]

        qt_jp1 = @views q[I_jp1, :]

        for i = (I_j + 1):(I_jp1-1)
            y[i, :] = A[i-1, :, : ] * y[i-1,:, :] + B[i-1, :, : ] * v[i-1,:, :] + s[i-1, :, :] # TODO check this y
            # q[i, :] = Theta[i, :, :] * y[i, :, :] + D[i, :, :] * qt_jp1 + beta[i, :, :] 
            # v[i, :] = Gamma[i, :, :] * y[i, :, :] + F[i, :, :] * qt_jp1 + gamma[i, :, :]
        end
    end

end

function PDP(N, n, m, P, I_separator, Q, R, S, B, A, z, r, s, s0)

    # TODO update N, n, m, P, I, and assert consistency
    Gamma = zeros(N+1, m, n)
    gamma = zeros(N+1, m)
    Theta = zeros(N+1, n, n)
    beta = zeros(N+1, n)
    D = zeros(N+1, n, n)
    F = zeros(N+1, m, n)

    Qt = zeros(P, n, n)
    Rt = zeros(P, n, m)
    St = zeros(P, m, m)
    Jt = zeros(P, n, n)

    At = zeros(P, n, n)
    Bt = zeros(P, n, m)

    st = zeros(P, n)
    tt = zeros(P, n)
    rt = zeros(P, m)

    Ah = zeros(P, n, n)
    Jh = zeros(P, n, n)
    Qh = zeros(P, n, n)
    sh = zeros(P, n)
    th = zeros(P, n)

    data = LQProbData(N, n, m, P, I_separator, Q, R, S, B, A, z, r, s, s0, Gamma, gamma, Theta, beta, D, F, Qt, Rt, St, Jt, At, Bt, st, tt, rt, Ah, Jh, Qh, sh, th)

    return data
end

end