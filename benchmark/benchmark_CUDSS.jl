using BlockStructuredSolvers
using CUDA
using BenchmarkTools
using LinearAlgebra
using Statistics
using Printf
using Random
using ProgressBars
using CUDSS
using SparseArrays

include("utils.jl")

function benchmark_with_cudss(m, n, P_start, level; num_trials::Int=5)
    println("\nBenchmarking with parameters:")
    println("m = $m (separator size)")
    println("n = $n (block size)")
    println("P_start = $P_start (initial separators)")
    println("level = $level (recursion level)")
    
    # Calculate total size
    P = P_start
    N = P * (m + 1) - m
    for i = 2:level
        P = N
        N = P * (m + 1) - m
    end
    println("Total size N = $N (total blocks, matrix size = $(N*n)×$(N*n))")
    
    # Arrays to store timing results
    our_factorize_times = Float64[]
    our_solve_times = Float64[]
    cudss_factorize_times = Float64[]
    cudss_solve_times = Float64[]
    
    # Progress bar for trials
    for trial in ProgressBars.tqdm(1:num_trials)
        # Generate test data
        A_list, B_list, x_list, x, d_list = generate_data(N, n)
        A_list_gpu, B_list_gpu, x_list_gpu, x_gpu, d_list_gpu = to_gpu(A_list, B_list, x_list, x, d_list)
        
        # Create the full sparse matrix for CUDSS
        BigMatrix, d_vec = construct_block_tridiagonal(A_list, B_list, d_list)
        
        # Convert to CUDA sparse format
        BigMatrix_cudss = CUDA.CUSPARSE.CuSparseMatrixCSR(BigMatrix)
        d_cudss = CuArray(d_vec)
        
        # Initialize our solvers
        data_gpu = initialize(P_start * (m + 1) - m, m, n, P_start, A_list_gpu, B_list_gpu, level)
        
        # Warmup runs (not timed)
        if trial == 1
            println("Performing warmup runs...")
            CUDA.@sync factorize!(data_gpu)
            CUDA.@sync solve!(data_gpu, d_list_gpu, x_gpu)
            CUDA.@sync chol_warmup = CUDSS.cholesky(BigMatrix_cudss; check=false)
            CUDA.@sync chol_warmup \ d_cudss
            GC.gc()
            CUDA.reclaim()
        end
        
        # Time our block Cholesky factorization
        CUDA.synchronize()
        start_time = time()
        CUDA.@sync factorize!(data_gpu)
        CUDA.synchronize()
        our_factorize_time = time() - start_time
        push!(our_factorize_times, our_factorize_time)
        
        # Time our block Cholesky solve
        CUDA.synchronize()
        start_time = time()
        CUDA.@sync solve!(data_gpu, d_list_gpu, x_gpu)
        CUDA.synchronize()
        our_solve_time = time() - start_time
        push!(our_solve_times, our_solve_time)
        
        # Time CUDSS factorization using events for precise timing
        start_event = CUDA.Event()
        stop_event = CUDA.Event()
        CUDA.record(start_event)
        chol = CUDSS.cholesky(BigMatrix_cudss; check=false)
        CUDA.record(stop_event)
        CUDA.synchronize()
        cudss_factorize_time = CUDA.time(start_event, stop_event) / 1000.0  # Convert to seconds
        push!(cudss_factorize_times, cudss_factorize_time)
        
        # Time CUDSS solve using events
        start_event = CUDA.Event()
        stop_event = CUDA.Event()
        CUDA.record(start_event)
        chol \ d_cudss
        CUDA.record(stop_event)
        CUDA.synchronize()
        cudss_solve_time = CUDA.time(start_event, stop_event) / 1000.0  # Convert to seconds
        push!(cudss_solve_times, cudss_solve_time)
        
        # Clean up to avoid memory leaks
        chol = nothing
        GC.gc()
        CUDA.reclaim()
    end
    
    # Calculate statistics
    our_factorize_mean = mean(our_factorize_times)
    our_factorize_std = std(our_factorize_times)
    our_solve_mean = mean(our_solve_times)
    our_solve_std = std(our_solve_times)
    
    cudss_factorize_mean = mean(cudss_factorize_times)
    cudss_factorize_std = std(cudss_factorize_times)
    cudss_solve_mean = mean(cudss_solve_times)
    cudss_solve_std = std(cudss_solve_times)
    
    factorize_speedup = cudss_factorize_mean / our_factorize_mean
    solve_speedup = cudss_solve_mean / our_solve_mean
    
    # Print detailed results
    println("\nDetailed Results (mean ± std over $num_trials trials):")
    println("Factorization:")
    @printf("  Our Method: %.4f ± %.4f seconds\n", our_factorize_mean, our_factorize_std)
    @printf("  CUDSS:      %.4f ± %.4f seconds\n", cudss_factorize_mean, cudss_factorize_std)
    if factorize_speedup > 1
        @printf("  Speedup:    %.2fx (our method is faster)\n", factorize_speedup)
    else
        @printf("  Speedup:    %.2fx (CUDSS is faster)\n", 1/factorize_speedup)
    end
    
    println("\nSolve:")
    @printf("  Our Method: %.4f ± %.4f seconds\n", our_solve_mean, our_solve_std)
    @printf("  CUDSS:      %.4f ± %.4f seconds\n", cudss_solve_mean, cudss_solve_std)
    if solve_speedup > 1
        @printf("  Speedup:    %.2fx (our method is faster)\n", solve_speedup)
    else
        @printf("  Speedup:    %.2fx (CUDSS is faster)\n", 1/solve_speedup)
    end
    
    return (
        our_factorize_mean=our_factorize_mean,
        our_factorize_std=our_factorize_std,
        our_solve_mean=our_solve_mean,
        our_solve_std=our_solve_std,
        cudss_factorize_mean=cudss_factorize_mean,
        cudss_factorize_std=cudss_factorize_std,
        cudss_solve_mean=cudss_solve_mean,
        cudss_solve_std=cudss_solve_std,
        factorize_speedup=factorize_speedup,
        solve_speedup=solve_speedup,
        N=N,
        n=n,
        m=m
    )
end

function run_benchmarks()
    println("Starting Block Cholesky vs CUDSS Benchmark Suite")
    println("===============================================")
    
    # Test cases with different parameters
    test_cases = [
        (2, 100, 3, 3),    # Small problem
        (4, 100, 3, 3),    # Medium problem
        (8, 100, 3, 3),    # Large problem
        (16, 100, 3, 2)    # Very large problem
    ]
    
    results = []
    
    for (m, n, P_start, level) in test_cases
        stats = benchmark_with_cudss(m, n, P_start, level)
        push!(results, stats)
    end
    
    # Print summary table
    println("\nBenchmark Summary")
    println("=================")
    println("Size    Our Factorize    CUDSS Factorize    Speedup    Our Solve    CUDSS Solve    Speedup")
    println("-----------------------------------------------------------------------------------------")
    for r in results
        matrix_size = r.N * r.n
        if r.factorize_speedup > 1
            factorize_speedup_str = @sprintf("%.2fx (ours)", r.factorize_speedup)
        else
            factorize_speedup_str = @sprintf("%.2fx (CUDSS)", 1/r.factorize_speedup)
        end
        
        if r.solve_speedup > 1
            solve_speedup_str = @sprintf("%.2fx (ours)", r.solve_speedup)
        else
            solve_speedup_str = @sprintf("%.2fx (CUDSS)", 1/r.solve_speedup)
        end
        
        @printf("%5d    %.4f±%.4f    %.4f±%.4f    %12s    %.4f±%.4f    %.4f±%.4f    %11s\n",
            matrix_size,
            r.our_factorize_mean, r.our_factorize_std,
            r.cudss_factorize_mean, r.cudss_factorize_std,
            factorize_speedup_str,
            r.our_solve_mean, r.our_solve_std,
            r.cudss_solve_mean, r.cudss_solve_std,
            solve_speedup_str
        )
    end
end

# Check if CUDA is available
if CUDA.functional()
    println("CUDA GPU detected. Starting benchmarks...")
    run_benchmarks()
else
    println("No CUDA GPU detected. Please ensure you have a CUDA-capable GPU and the necessary drivers installed.")
end 