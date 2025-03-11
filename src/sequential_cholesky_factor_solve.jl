struct BlockTriDiagData_sequential_cholesky_factor{
    T, 
    MT <: AbstractArray{T, 3}}

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

function solve_sequential_cholesky_factor!(data::BlockTriDiagData_sequential_cholesky_factor, d, x)

    N = data.N
    n = data.n
    A_list = data.A_list
    B_list = data.B_list

    cholesky_solve!(A_list, B_list, d, N, n)

    x .= d

    return nothing
end