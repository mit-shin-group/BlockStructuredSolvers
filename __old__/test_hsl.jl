using HSL_jll
using HSL
using LinearAlgebra
using SparseArrays

function LIBHSL_isfunctional()
    @ccall libhsl.LIBHSL_isfunctional()::Bool
end
bool = LIBHSL_isfunctional()

display(bool)

A = sprand(100, 100, 0.1)
b = rand(100)
x = ma97_solve(A, b) # 57 27 97 

display(x)