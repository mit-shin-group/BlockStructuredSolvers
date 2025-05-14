for (Xpotrf_buffer, Xpotrf, Xtrsm, Xgemm, T) in (
    (:cusolverDnSpotrf_bufferSize, :cusolverDnSpotrf, :cublasStrsm_v2, :cublasSgemm_v2, :Float32),
    (:cusolverDnDpotrf_bufferSize, :cusolverDnDpotrf, :cublasDtrsm_v2, :cublasDgemm_v2, :Float64),
)
    @eval begin
        function cholesky_factorize!(A_ptrs::CuVector{<:CuPtr{$T}}, B_ptrs::CuVector{<:CuPtr{$T}}, N, n)

            #TODO move this out
            dh = CUSOLVER.dense_handle()

            function bufferSize()
                out = Ref{Cint}(0)
                CUSOLVER.$Xpotrf_buffer(dh, CUBLAS.CUBLAS_FILL_MODE_UPPER, n, zeros($T, n, n), n, out)
                out[] * sizeof($T)
            end

            CUDA.with_workspace(dh.workspace_gpu, bufferSize) do buffer
                CUSOLVER.$Xpotrf(dh, CUBLAS.CUBLAS_FILL_MODE_UPPER, n, A_ptrs[1], n,
                    buffer, sizeof(buffer) ÷ sizeof($T), dh.info)
                for i = 2:N
                    CUBLAS.$Xtrsm(CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                        CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_DIAG_NON_UNIT, n, n, one($T), A_ptrs[i-1], n, B_ptrs[i-1], n)
                    CUBLAS.$Xgemm(CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
                        n, n, n, -one($T), B_ptrs[i-1], n, B_ptrs[i-1], n, one($T), A_ptrs[i], n)
                    CUSOLVER.$Xpotrf(dh, CUBLAS.CUBLAS_FILL_MODE_UPPER, n, A_ptrs[i], n,
                        buffer, sizeof(buffer) ÷ sizeof($T), dh.info)
                end
            end
        end

        function cholesky_solve!(A_ptrs::CuVector{<:CuPtr{$T}}, B_ptrs::CuVector{<:CuPtr{$T}}, d_ptrs::CuVector{<:CuPtr{$T}}, N, n, nd)

            CUBLAS.$Xtrsm(CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_DIAG_NON_UNIT, n, nd, one($T), A_ptrs[1], n, d_ptrs[1], n)
        
            for i = 2:N
                CUBLAS.$Xgemm(CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
                    n, nd, n, -one($T), B_ptrs[i-1], n, d_ptrs[i-1], n, one($T), d_ptrs[i], n)
                CUBLAS.$Xtrsm(CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                    CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_DIAG_NON_UNIT, n, nd, one($T), A_ptrs[i], n, d_ptrs[i], n)
            end
        
            CUBLAS.$Xtrsm(CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_DIAG_NON_UNIT, n, nd, one($T), A_ptrs[N], n, d_ptrs[N], n)
        
            for i = N-1:-1:1
                CUBLAS.$Xgemm(CUBLAS.handle(), CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_OP_N,
                    n, nd, n, -one($T), B_ptrs[i], n, d_ptrs[i+1], n, one($T), d_ptrs[i], n)
                CUBLAS.$Xtrsm(CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                    CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_DIAG_NON_UNIT, n, nd, one($T), A_ptrs[i], n, d_ptrs[i], n)
            end
        
        end
    end
end

for (XpotrfBatched, XtrsmBatched, XgemmBatched, T) in (
    (:cusolverDnSpotrfBatched, :cublasStrsmBatched_64, :cublasSgemmBatched_64, :Float32),
    (:cusolverDnDpotrfBatched, :cublasDtrsmBatched_64, :cublasDgemmBatched_64, :Float64),
)
    @eval begin
        function cholesky_factorize_batched!(A_ptrs::CuVector{<:CuPtr{$T}}, B_ptrs::CuVector{<:CuPtr{$T}}, bsz, sep, start, N, n)

            # ---- one info buffer and one cuSOLVER handle ---------------------------
            dh = CUSOLVER.dense_handle()
            CUDA.resize!(dh.info, bsz)

            # ---- first block row ----------------------------------------------------
            CUSOLVER.$XpotrfBatched(
                dh, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                n, A_ptrs[start:sep:end], n, dh.info, bsz)
            
            # ---- remaining block rows ----------------------------------------------
            for i in (start+1):(start+N-1)

                CUBLAS.$XtrsmBatched(
                    CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                    CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_DIAG_NON_UNIT,
                    n, n, one($T), A_ptrs[i-1:sep:end], n, B_ptrs[(i-1):sep:end], n, bsz) #TODO check : or use view or unsafe_wrap

                CUBLAS.$XgemmBatched(
                    CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
                    n, n, n, -one($T),
                    B_ptrs[(i-1):sep:end], n, B_ptrs[(i-1):sep:end], n,
                    one($T), A_ptrs[i:sep:end], n, bsz)

                CUSOLVER.$XpotrfBatched(
                    dh, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                    n, A_ptrs[i:sep:end], n, dh.info, bsz)

            end
            
        end

        function cholesky_solve_batched!(A_ptrs::CuVector{<:CuPtr{$T}}, B_ptrs::CuVector{<:CuPtr{$T}}, d_ptrs::CuVector{<:CuPtr{$T}}, bsz, sep, start, N, n, nd)

            CUBLAS.$XtrsmBatched(
                CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_DIAG_NON_UNIT,
                n, nd, one($T), A_ptrs[start:sep:end], n, d_ptrs[start:sep:end], n, bsz)

            for i = (start+1):(start+N-1)
                CUBLAS.$XgemmBatched(
                    CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
                    n, nd, n, -one($T),
                    B_ptrs[(i-1):sep:end], n, d_ptrs[(i-1):sep:end], n,
                    one($T), d_ptrs[i:sep:end], n, bsz)

                CUBLAS.$XtrsmBatched(
                    CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                    CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_DIAG_NON_UNIT,
                    n, nd, one($T), A_ptrs[i:sep:end], n, d_ptrs[i:sep:end], n, bsz)
            end

            for i = (start+N-1):-1:(start+1)
                CUBLAS.$XtrsmBatched(
                    CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                    CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_DIAG_NON_UNIT,
                    n, nd, one($T), A_ptrs[i:sep:end], n, d_ptrs[i:sep:end], n, bsz)

                CUBLAS.$XgemmBatched(
                    CUBLAS.handle(), CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_OP_N,
                    n, nd, n, -one($T),
                    B_ptrs[(i-1):sep:end], n, d_ptrs[i:sep:end], n,
                    one($T), d_ptrs[(i-1):sep:end], n, bsz)
            end

            CUBLAS.$XtrsmBatched(
                CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
                CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_DIAG_NON_UNIT,
                n, nd, one($T), A_ptrs[start:sep:end], n, d_ptrs[start:sep:end], n, bsz)    

        end

        function compute_schur_complement!(G_ptrs::CuVector{<:CuPtr{$T}}, F_ptrs::CuVector{<:CuPtr{$T}}, M_2n_ptrs::CuVector{<:CuPtr{$T}}, LHS_A_tensor::CuArray{<:$T, 3}, LHS_B_tensor::CuArray{<:$T, 3}, M_2n_tensor::CuArray{<:$T, 3}, P, m, n)

            for j = 2:m+1
                CUBLAS.$XgemmBatched(
                    CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
                    2*n, 2*n, n, one($T),
                    G_ptrs[j:(m+1):end], n, F_ptrs[j:(m+1):end], n,
                    one($T), M_2n_ptrs, 2*n, P-1)
            end
        
            view(LHS_A_tensor, :, :, 1:P-1) .-= view(M_2n_tensor, 1:n, 1:n, :);
            view(LHS_A_tensor, :, :, 2:P) .-= view(M_2n_tensor, n+1:2*n, n+1:2*n, :);
            LHS_B_tensor .-= view(M_2n_tensor, 1:n, n+1:2*n, :);
        
        end

        function compute_schur_RHS!(F_ptrs::CuVector{<:CuPtr{$T}}, d_ptrs::CuVector{<:CuPtr{$T}}, b_ptrs::CuVector{<:CuPtr{$T}}, d_tensor::CuArray{<:$T, 3}, b_tensor::CuArray{<:$T, 3}, N, n, m, P)
            
            for i = 2:m+1
                CUBLAS.$XgemmBatched(
                    CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
                    2*n, 1, n, -one($T),
                    F_ptrs[i:(m+1):end], n, d_ptrs[i:(m+1):end], n,
                    one($T), b_ptrs, 2*n, P-1)
            end
            
            view(d_tensor, :, :, 1:(m+1):N-1) .+= view(b_tensor, 1:n, :, :);
            view(d_tensor, :, :, (m+2):(m+1):N) .+= view(b_tensor, n+1:2*n, :, :);
        end

        function update_boundary!(M_ptrs_1::CuVector{<:CuPtr{$T}}, M_ptrs_2::CuVector{<:CuPtr{$T}}, d_ptrs::CuVector{<:CuPtr{$T}}, P, n, m)
            CUBLAS.$XgemmBatched(
                CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
                n, 1, n, -one($T),
                M_ptrs_1, n, d_ptrs[1:(m+1):end-1], n,
                one($T), d_ptrs[2:(m+1):end], n, P-1)
            CUBLAS.$XgemmBatched(
                CUBLAS.handle(), CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_OP_N,
                n, 1, n, -one($T),
                M_ptrs_2, n, d_ptrs[(m+2):(m+1):end], n,
                one($T), d_ptrs[(m+1):(m+1):end], n, P-1)
        end

    end
end