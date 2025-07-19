struct BlockTriDiagData_cpu{T}

    N::Int
    n::Int

    A_list::Vector{Array{T, 2}}
    B_list::Vector{Array{T, 2}}
    d_list::Vector{Array{T, 2}}

end

function initialize_cpu(N, n, ::Type{T}) where T

    A_list = [Array{T, 2}(zeros(n, n)) for i in 1:N]
    B_list = [Array{T, 2}(zeros(n, n)) for i in 1:N-1]
    d_list = [Array{T, 2}(zeros(n, 1)) for i in 1:N]

    data = BlockTriDiagData_cpu(
        N,
        n,
        A_list,
        B_list,
        d_list
    )

    return data

end

function factorize!(data::BlockTriDiagData_cpu)

    N = data.N

    A_list = data.A_list
    B_list = data.B_list

    cholesky_factorize!(A_list, B_list, N)

end

function solve!(data::BlockTriDiagData_cpu)

    N = data.N
    A_list = data.A_list
    B_list = data.B_list
    d_list = data.d_list
    
    cholesky_solve!(A_list, B_list, d_list, N)

end