#TODO might not need CUDA potrf/... anymore due to batched implementation
function mypotrf!(uplo::Char, A::StridedCuMatrix{T}) where {T}

    cupotrf!(uplo, A)

end 

function mypotrf!(uplo::Char, A::AbstractMatrix{T}) where {T}

    lapotrf!(uplo, A)

end

function mygemm!(transA::Char, transB::Char, alpha::Number, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}, beta::Number, C::StridedCuMatrix{T}) where {T}

    cugemm!(transA, transB, alpha, A, B, beta, C)

end

function mygemm!(transA::Char, transB::Char, alpha::Number, A::AbstractMatrix{T}, B::AbstractMatrix{T}, beta::Number, C::AbstractMatrix{T}) where {T}

    lagemm!(transA, transB, alpha, A, B, beta, C)

end

function mytrsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Number, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where {T}

    cutrsm!(side, uplo, transa, diag, alpha, A, B)

end

function mytrsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Number, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}

    latrsm!(side, uplo, transa, diag, alpha, A, B)

end

function _bss_norm(A::Vector{<:CuMatrix{T}}) where {T}

    cunorm(A)

end

function _bss_norm(A::Vector{<:AbstractMatrix{T}}) where {T}

    lanorm(A)

end