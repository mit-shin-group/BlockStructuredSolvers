using AMDGPU
using AMDGPU.rocSOLVER, AMDGPU.rocBLAS

for (Xpotrf, Xtrsm, Xgemm, T) in (
    (:rocsolver_spotrf, :rocblas_strsm, :rocblas_sgemm, :Float32),
    (:rocsolver_dpotrf, :rocblas_dtrsm, :rocblas_dgemm, :Float64),
)
    @eval begin
        function cholesky_factorize!(A_ptrs::ROCVector{<:Ptr{$T}}, B_ptrs::ROCVector{<:Ptr{$T}}, N, n)
            dh = rocBLAS.handle()
            dinfo = ROCVector{Cint}(undef, 1)
            rocSOLVER.$Xpotrf(dh, rocBLAS.rocblas_fill_upper, n, A_ptrs[1], n, dinfo)
            for i = 2:N
                rocBLAS.$Xtrsm(dh, rocBLAS.rocblas_side_left, rocBLAS.rocblas_fill_upper,
                               rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_diagonal_non_unit,
                               n, n, one($T), A_ptrs[i-1], n, B_ptrs[i-1], n)
                rocBLAS.$Xgemm(dh, rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_operation_none,
                               n, n, n, -one($T), B_ptrs[i-1], n, B_ptrs[i-1], n, one($T), A_ptrs[i], n)
                rocBLAS.$Xpotrf(dh, rocBLAS.rocblas_fill_upper, n, A_ptrs[i], n, dinfo)
            end
        end

        function cholesky_solve!(A_ptrs::ROCVector{<:Ptr{$T}}, B_ptrs::ROCVector{<:Ptr{$T}}, d_ptrs::ROCVector{<:Ptr{$T}}, N, n, nd)
            dh = rocBLAS.handle()
            rocBLAS.$Xtrsm(dh, rocBLAS.rocblas_side_left, rocBLAS.rocblas_fill_upper,
                           rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_diagonal_non_unit,
                           n, nd, one($T), A_ptrs[1], n, d_ptrs[1], n)

            for i = 2:N
                rocBLAS.$Xgemm(dh, rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_operation_none,
                               n, nd, n, -one($T), B_ptrs[i-1], n, d_ptrs[i-1], n, one($T), d_ptrs[i], n)
                rocBLAS.$Xtrsm(dh, rocBLAS.rocblas_side_left, rocBLAS.rocblas_fill_upper,
                               rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_diagonal_non_unit,
                               n, nd, one($T), A_ptrs[i], n, d_ptrs[i], n)
            end

            rocBLAS.$Xtrsm(dh, rocBLAS.rocblas_side_left, rocBLAS.rocblas_fill_upper,
                           rocBLAS.rocblas_operation_none, rocBLAS.rocblas_diagonal_non_unit,
                           n, nd, one($T), A_ptrs[N], n, d_ptrs[N], n)

            for i = N-1:-1:1
                rocBLAS.$Xgemm(dh, rocBLAS.rocblas_operation_none, rocBLAS.rocblas_operation_none,
                               n, nd, n, -one($T), B_ptrs[i], n, d_ptrs[i+1], n, one($T), d_ptrs[i], n)
                rocBLAS.$Xtrsm(dh, rocBLAS.rocblas_side_left, rocBLAS.rocblas_fill_upper,
                               rocBLAS.rocblas_operation_none, rocBLAS.rocblas_diagonal_non_unit, n, nd,
                               one($T), A_ptrs[i], n, d_ptrs[i], n)
            end
        end
    end
end

for (XpotrfBatched, XtrsmBatched, XgemmBatched, T) in (
    (:rocsolver_spotrf_batched, :rocblas_strsm_batched, :rocblas_sgemm_batched, :Float32),
    (:rocsolver_dpotrf_batched, :rocblas_dtrsm_batched, :rocblas_dgemm_batched, :Float64),
)
    @eval begin
        function cholesky_factorize_batched!(A_ptrs::ROCVector{<:Ptr{$T}}, B_ptrs::ROCVector{<:Ptr{$T}}, bsz, sep, start, N, n)
            dh = rocBLAS.handle()
            dinfo = ROCVector{Cint}(undef, 1)

            rocSOLVER.$XpotrfBatched(dh, rocBLAS.rocblas_fill_upper, n, A_ptrs[start:sep:end], n, dinfo, bsz)

            for i in (start+1):(start+N-1)

                rocBLAS.$XtrsmBatched(
                    dh, rocBLAS.rocblas_side_left, rocBLAS.rocblas_fill_upper,
                    rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_diagonal_non_unit,
                    n, n, one($T), A_ptrs[i-1:sep:end], n, B_ptrs[(i-1):sep:end], n, bsz)

                rocBLAS.$XgemmBatched(
                    dh, rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_operation_none,
                    n, n, n, -one($T), B_ptrs[(i-1):sep:end], n, B_ptrs[(i-1):sep:end], n,
                    one($T), A_ptrs[i:sep:end], n, bsz)

                rocSOLVER.$XpotrfBatched(dh, rocBLAS.rocblas_fill_upper, n, A_ptrs[i:sep:end], n, dinfo, bsz)
            end
        end

        function cholesky_solve_batched!(A_ptrs::ROCVector{<:Ptr{$T}}, B_ptrs::ROCVector{<:Ptr{$T}}, d_ptrs::ROCVector{<:Ptr{$T}}, bsz, sep, start, N, n, nd)
            dh = rocBLAS.handle()

            rocBLAS.$XtrsmBatched(
                dh, rocBLAS.rocblas_side_left, rocBLAS.rocblas_fill_upper,
                rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_diagonal_non_unit,
                n, nd, one($T), A_ptrs[start:sep:end], n, d_ptrs[start:sep:end], n, bsz)

            for i = (start+1):(start+N-1)
                rocBLAS.$XgemmBatched(
                    dh, rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_operation_none,
                    n, nd, n, -one($T), B_ptrs[(i-1):sep:end], n, d_ptrs[(i-1):sep:end], n,
                    one($T), d_ptrs[i:sep:end], n, bsz)

                rocBLAS.$XtrsmBatched(
                    dh, rocBLAS.rocblas_side_left, rocBLAS.rocblas_fill_upper,
                    rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_diagonal_non_unit,
                    n, nd, one($T), A_ptrs[i:sep:end], n, d_ptrs[i:sep:end], n, bsz)
            end

            for i = (start+N-1):-1:(start+1)
                rocBLAS.$XtrsmBatched(
                    dh, rocBLAS.rocblas_side_left, rocBLAS.rocblas_fill_upper,
                    rocBLAS.rocblas_operation_none, rocBLAS.rocblas_diagonal_non_unit,
                    n, nd, one($T), A_ptrs[i:sep:end], n, d_ptrs[i:sep:end], n, bsz)

                rocBLAS.$XgemmBatched(
                    dh, rocBLAS.rocblas_operation_none, rocBLAS.rocblas_operation_none,
                    n, nd, n, -one($T), B_ptrs[(i-1):sep:end], n, d_ptrs[i:sep:end], n,
                    one($T), d_ptrs[(i-1):sep:end], n, bsz)
            end

            rocBLAS.$XtrsmBatched(
                dh, rocBLAS.rocblas_side_left, rocBLAS.rocblas_fill_upper,
                rocBLAS.rocblas_operation_none, rocBLAS.rocblas_diagonal_non_unit,
                n, nd, one($T), A_ptrs[start:sep:end], n, d_ptrs[start:sep:end], n, bsz)    
        end
    end
end
