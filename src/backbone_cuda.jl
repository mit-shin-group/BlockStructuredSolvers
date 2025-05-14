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
    (:cusolverDnSpotrfBatched, :cublasStrsmBatched, :cublasSgemmBatched, :Float32),
    (:cusolverDnDpotrfBatched, :cublasDtrsmBatched, :cublasDgemmBatched, :Float64),
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

    end
end