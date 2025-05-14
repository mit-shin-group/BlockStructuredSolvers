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

        function compute_schur_complement!(G_ptrs::ROCVector{<:Ptr{$T}}, F_ptrs::ROCVector{<:Ptr{$T}}, M_2n_ptrs::ROCVector{<:Ptr{$T}}, LHS_A_tensor::ROCArray{<:$T, 3}, LHS_B_tensor::ROCArray{<:$T, 3}, M_2n_tensor::ROCArray{<:$T, 3}, P, m, n)

            dh = rocBLAS.handle()

            for j = 2:m+1
                rocBLAS.$XgemmBatched(
                    dh, rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_operation_none,
                    2*n, 2*n, n, one($T),
                    G_ptrs[j:(m+1):end], n, F_ptrs[j:(m+1):end], n,
                    one($T), M_2n_ptrs, 2*n, P-1)
            end
        
            view(LHS_A_tensor, :, :, 1:P-1) .-= view(M_2n_tensor, 1:n, 1:n, :);
            view(LHS_A_tensor, :, :, 2:P) .-= view(M_2n_tensor, n+1:2*n, n+1:2*n, :);
            LHS_B_tensor .-= view(M_2n_tensor, 1:n, n+1:2*n, :);
        
        end

        function compute_schur_RHS!(F_ptrs::ROCVector{<:Ptr{$T}}, d_ptrs::ROCVector{<:Ptr{$T}}, b_ptrs::ROCVector{<:Ptr{$T}}, d_tensor::ROCArray{<:$T, 3}, b_tensor::ROCArray{<:$T, 3}, N, n, m, P)
            
            dh = rocBLAS.handle()

            for i = 2:m+1
                rocBLAS.$XgemmBatched(
                    dh, rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_operation_none,
                    2*n, 1, n, -one($T),
                    F_ptrs[i:(m+1):end], n, d_ptrs[i:(m+1):end], n,
                    one($T), b_ptrs, 2*n, P-1)
            end
            
            view(d_tensor, :, :, 1:(m+1):N-1) .+= view(b_tensor, 1:n, :, :);
            view(d_tensor, :, :, (m+2):(m+1):N) .+= view(b_tensor, n+1:2*n, :, :);
        end

        function update_boundary!(M_ptrs_1::ROCVector{<:Ptr{$T}}, M_ptrs_2::ROCVector{<:Ptr{$T}}, d_ptrs::ROCVector{<:Ptr{$T}}, P, n, m)
            
            dh = rocBLAS.handle()
            
            rocBLAS.$XgemmBatched(
                dh, rocBLAS.rocblas_operation_transpose, rocBLAS.rocblas_operation_none,
                n, 1, n, -one($T),
                M_ptrs_1, n, d_ptrs[1:(m+1):end-1], n,
                one($T), d_ptrs[2:(m+1):end], n, P-1)
            rocBLAS.$XgemmBatched(
                dh, rocBLAS.rocblas_operation_none, rocBLAS.rocblas_operation_none,
                n, 1, n, -one($T),
                M_ptrs_2, n, d_ptrs[(m+2):(m+1):end], n,
                one($T), d_ptrs[(m+1):(m+1):end], n, P-1)
        end


    end
end
