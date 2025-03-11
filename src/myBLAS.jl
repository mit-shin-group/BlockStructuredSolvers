function mypotrf!(uplo::Char, A::StridedCuMatrix{T}) where {T}

    cupotrf!(uplo, A)

end 

function mypotrf!(uplo::Char, A::AbstractMatrix{T}) where {T}

    lapotrf!(uplo, A)

end

function mygemv!(trans::Char, alpha::Number, A::StridedCuMatrix{T}, x::StridedCuVector{T}, beta::Number, y::StridedCuVector{T}) where {T}

    cugemv!(trans, alpha, A, x, beta, y)

end

function mygemv!(trans::Char, alpha::Number, A::AbstractMatrix{T}, x::AbstractVector{T}, beta::Number, y::AbstractVector{T}) where {T}

    lagemv!(trans, alpha, A, x, beta, y)

end

function mygemm!(transA::Char, transB::Char, alpha::Number, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}, beta::Number, C::StridedCuMatrix{T}) where {T}

    cugemm!(transA, transB, alpha, A, B, beta, C)

end

function mygemm!(transA::Char, transB::Char, alpha::Number, A::AbstractMatrix{T}, B::AbstractMatrix{T}, beta::Number, C::AbstractMatrix{T}) where {T}

    lagemm!(transA, transB, alpha, A, B, beta, C)

end

function mytrsv!(uplo::Char, trans::Char, diag::Char, A::StridedCuMatrix{T}, x::StridedCuVector{T}) where {T}

    cutrsv!(uplo, trans, diag, A, x)

end

function mytrsv!(uplo::Char, trans::Char, diag::Char, A::AbstractMatrix{T}, x::AbstractVector{T}) where {T}

    latrsv!(uplo, trans, diag, A, x)

end

function mytrsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Number, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where {T}

    cutrsm!(side, uplo, transa, diag, alpha, A, B)

end

function mytrsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Number, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}

    latrsm!(side, uplo, transa, diag, alpha, A, B)

end