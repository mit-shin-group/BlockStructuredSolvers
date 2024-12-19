using LinearAlgebra

N = 8

P = trunc(Int, (N + 1) / 2)

I_separator = [1]

for j = 1:P-1
    push!(I_separator, I_separator[end]+2)
end

push!(I_separator, N+1)

Q = rand(N+1, 2, 2)
R = rand(N+1, 2, 2)
S = rand(N+1, 2, 2)

A = rand(N+1, 2, 2)
B = rand(N+1, 2, 2)

z = rand(N+1, 2, 1)
r = rand(N+1, 2, 1)
s = rand(N+1, 2, 1)

### Start of Algorithm

Gamma = zeros(N+1, 2, 2)
gamma = zeros(N+1, 2, 1)
theta = zeros(N+1, 2, 2)
beta = zeros(N+1, 2, 1)
D = zeros(N+1, 2, 2)
F = zeros(N+1, 2, 2)

for j = 1:P
    
    I_jp1 = I_separator[j+1]
    I_j = I_separator[j]

    theta[I_jp1, :, :] = zeros(2, 2)
    beta[I_jp1, : ,:] = zeros(2, 1)
    D[I_jp1, :, :] = Diagonal([1, 1])

    i = I_jp1 - 1
    while i >= I_j+1 #TODO

        Q_i = Q[i, :, :]
        R_i = R[i, :, :]
        S_i = S[i, :, :]

        A_i = A[i, :, :]
        B_i = B[i, :, :]
        A_i_T = transpose(A_i)
        B_i_T = transpose(B_i)
        
        z_i = z[i, :, :]
        r_i = r[i, :, :]
        s_i = s[i, :, :]

        theta_ip1 = theta[i+1, :, :]
        beta_ip1 = beta[i+1, :, :]
        D_ip1 = D[i+1, :, :]

        Gamma_i = - inv(S_i + B_i_T * theta_ip1 * B_i) * (transpose(R_i) + B_i_T * theta_ip1 * A_i)
        gamma_i = - inv(S_i + B_i_T * theta_ip1 * B_i) * (r_i + B_i_T * (theta_ip1 * s_i + beta_ip1))
        F_i = inv(S_i + B_i_T * theta_ip1 * B_i) * B_i_T * D_ip1
        D_i = transpose(A_i + B_i * Gamma_i) * D_ip1
        theta_i = (Q_i + A_i_T * theta_ip1 * A_i) + (R_i + A_i_T * theta_ip1 * B_i) * Gamma_i
        beta_i = A_i_T * theta_ip1 * (B_i * gamma_i + s_i) + A_i_T * beta_ip1 + z_i + R_i * gamma_i

        Gamma[i, :, :] = Gamma_i
        gamma[i, :, :] = gamma_i
        F[i, :, :] = F_i
        D[i, :, :] = D_i
        theta[i, :, :] = theta_i
        beta[i, :, :] = beta_i

        i -= 1
    end

end

Qt = zeros(P, 2, 2)
Rt = zeros(P, 2, 2)
St = zeros(P, 2, 2)
Jt = zeros(P, 2, 2)

At = zeros(P, 2, 2)
Bt = zeros(P, 2, 2)

st = zeros(P, 2, 1)
tt = zeros(P, 2, 1)
rt = zeros(P, 2, 1)

for j = 1:P

    I_jp1 = I_separator[j+1]
    I_j = I_separator[j]

    At[j, :, :] = transpose(D[I_jp1, :, :]) * A[I_j, :, :]
    Bt[j, :, :] = transpose(D[I_jp1, :, :]) * B[I_j, :, :]

    J_temp = zeros(2, 2)
    s_temp = zeros(2, 1)
    for i = (I_j + 1):(I_jp1-1)
        J_temp += transpose(D[i+1, :, :]) * B[i, :, :] * F[i, :, :]
        s_temp += transpose(D[i+1, :, :]) * (B[i, :, :] * gamma[i, :, :] + s[i, :, :]) + transpose(D[I_jp1, :, :]) * s[I_jp1,:, :]
    end
    Jt[j, :, :] = J_temp
    st[j, :, :] = s_temp

    tt[j, :, :] = z[I_j, :, :] + transpose(A[I_j, :, :]) * (theta[I_jp1, :, :] * s[I_j, :, :] + beta[I_jp1, :, :])
    Rt[j, :, :] = R[I_j, :, :] + transpose(A[I_j, :, :]) * theta[I_jp1, :, :] * B[I_j, :, :] 
    Qt[j, :, :] = Q[I_j, :, :] + transpose(A[I_j, :, :]) * theta[I_jp1, :, :] * A[I_j, :, :] 

    rt[j, :, :] = r[I_j, :, :] + transpose(B[I_j, :, :])  * (theta[I_jp1, :, :] * s[I_j, :, :] + beta[I_jp1, :, :])
    St[j, :, :] = S[I_j, :, :] + transpose(B[I_j, :, :]) * theta[I_jp1, :, :] * B[I_j, :, :] 
    
end

Coeff = zeros(2 * (3 * P + 2), 2 * (3 * P + 2))

for j = 1:P
    Coeff[2*3*(j-1)+1:2*3*(j-1)+2, 2*3*(j-1)+3:2*3*(j-1)+4] = - Diagonal([1, 1])
    Coeff[2*3*(j-1)+3:2*3*(j-1)+4, 2*3*(j-1)+1:2*3*(j-1)+2] = - Diagonal([1, 1])
    Coeff[2*3*(j-1)+3:2*3*(j-1)+8, 2*3*(j-1)+3:2*3*(j-1)+8] = [Qt[j, :, :] Qt[j, :, :] Qt[j, :, :]; Qt[j, :, :] Qt[j, :, :] Qt[j, :, :]; Qt[j, :, :] Qt[j, :, :] Qt[j, :, :]]
end

Coeff[2*3*P+1:2*3*P+2, 2*3*P+3:2*3*P+4] = - Diagonal([1, 1])
Coeff[2*3*P+3:2*3*P+4, 2*3*P+1:2*3*P+2] = - Diagonal([1, 1])
Coeff[end-1:end, end-1:end] = Q[end, :, :]

s_0 = [0; 0] #TODO what is s0

RHS = s_0
for j=1:P
    global RHS = vcat(RHS, -tt[j, :, :], -rt[j, :, :], -st[j, :, :])
end

RHS = vcat(RHS, -z[end, :, :])
sol = Coeff \ RHS

q = zeros(N+1, 2, 1)
y = zeros(N+1, 2, 1)
v = zeros(N+1, 2, 1)

q[N+1, :, :] = sol[end-3:end-2]
y[N+1, :, :] = sol[end-1:end]

for j = P:-1:1

    I_jp1 = I_separator[j+1]
    I_j = I_separator[j]

    q[I_j, :, :] = sol[(j-1)*6+1:(j-1)*6+2]
    y[I_j, :, :] = sol[(j-1)*6+3:(j-1)*6+4]
    v[I_j, :, :] = sol[(j-1)*6+5:(j-1)*6+6]
    qt_jp1 = q[I_jp1, :, :]

    for i = (I_j + 1):(I_jp1-1)
        q[i, :, :] = theta[i, :, :] * y[i, :, :] + D[i, :, :] * qt_jp1 + beta[i, :, :] # TODO what is y
        v[i, :, :] = Gamma[i, :, :] * y[i, :, :] + F[i, :, :] * qt_jp1 + gamma[i, :, :]
    end
end

display(y)