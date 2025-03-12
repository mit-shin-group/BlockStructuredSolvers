struct BlockTriDiagData_sequential_cholesky_factor{
    T, 
    MT <:Vector{<:AbstractMatrix{T}}}

    N::Int
    n::Int

    A_list::MT
    B_list::MT

end

function initialize_sequential_cholesky_factor(N, n, A_list, B_list)

    data = BlockTriDiagData_sequential_cholesky_factor(
        N,
        n,
        A_list,
        B_list
    )

    return data

end

function factorize_sequential_cholesky_factor!(
    data::BlockTriDiagData_sequential_cholesky_factor
)

N = data.N

A_list = data.A_list
B_list = data.B_list

cholesky_factorize!(A_list, B_list, N)

end

function solve_sequential_cholesky_factor!(data::BlockTriDiagData_sequential_cholesky_factor, d_list, x)

    N = data.N
    n = data.n
    A_list = data.A_list
    B_list = data.B_list

    cholesky_solve!(A_list, B_list, d_list, N, n)

    x .= d_list

    return nothing
end