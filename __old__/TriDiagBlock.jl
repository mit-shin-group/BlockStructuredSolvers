module TriDiagBlock

using LinearAlgebra

export TriDiagBlockData, ThomasSolve

mutable struct TriDiagBlockData{T, MT <: AbstractArray{T,3}} # TODO preallocate all the struct

    N::Int

    A_list::MT
    B_list::MT

    temp_list::MT

end

function ThomasFactorize(
    data::TriDiagBlockData
    )

    N = data.N

    A_list = data.A_list
    B_list = data.B_list
    temp_list = data.temp_list

    for i = 2:N

        temp_list[i, :, :] =  B_list[i-1, :, :] * inv(A_list[i-1, :, :]) # potri!
        A_list[i, :, :] = A_list[i, :, :] - temp_list[i, :, :] * B_list[i-1, :, :]

    end

end

function ThomasSolve(
    data::TriDiagBlockData, 
    d_list, 
    x_list
    )

        N = data.N
        A_list = data.A_list
        B_list = data.B_list
        temp_list = data.temp_list

        for i = 2:N

            d_list[i, :] =  d_list[i, :] - temp_list[i, :, :] * d_list[i-1, :]

        end

        x_list[N, :] = inv(A_list[N, :, :]) * d_list[N, :] #TODO LAPACK

        for j = N-1:-1:1

            x_list[j, :] = inv(A_list[j, :, :]) * (d_list[j, :] - B_list[j, :, :]' * x_list[j+1, :])

        end

    end

end