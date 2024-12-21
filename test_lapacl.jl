using LinearAlgebra

temp = rand(Float64, 100, 100);
A = temp * temp';
AA = copy(A)

# cholesky!(A)
LAPACK.potrf!('U', A)
LAPACK.potri!('U', A)

# A - inv(AA)