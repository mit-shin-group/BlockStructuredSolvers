using LinearAlgebra, SparseArrays
using CUDA
using AMDGPU
using Random
using CUDSS
using BlockStructuredSolvers

import Pkg
include("utils.jl")

using Printf, ProgressBars, Statistics, Dates

if CUDA.functional()
    device!(0)
end

######

seed = 42 # random seed for reproducibility
T = Float64

# Determine GPU backend
if AMDGPU.functional()
    M = ROCArray
else
    M = CuArray
end

# Helper functions for cleaner code
function gpu_elapsed(f, backend_type)
    if backend_type == CuArray
        return CUDA.@elapsed f()
    else
        return AMDGPU.@elapsed f()
    end
end

function gpu_sync(f, backend_type)
    if backend_type == CuArray
        return CUDA.@sync f()
    else
        return AMDGPU.@sync f()
    end
end

function gpu_allowscalar(f, backend_type)
    if backend_type == CuArray
        @CUDA.allowscalar f()
    else
        @AMDGPU.allowscalar f()
    end
end

function gpu_reclaim(backend_type)
    GC.gc()
    if backend_type == CuArray
        CUDA.reclaim()
    else
        AMDGPU.GC.gc()
    end
end

# --- Kalman Filter Problem Generation ---
function generate_kalman_filter_problem(n_state, m_obs, N_timesteps, sparsity_G=1.0, sparsity_H=1.0)
    """
    Generate a Kalman Filter problem that results in a block tridiagonal system.
    
    Parameters:
    - n_state: State dimension (n)
    - m_obs: Observation dimension (m)
    - N_timesteps: Number of time steps (N)
    - sparsity_G: Sparsity level for state transition matrix G_k
    - sparsity_H: Sparsity level for observation matrix H_k
    
    Returns:
    - A: Block tridiagonal system matrix (SPD)
    - b: Right-hand side vector
    - block_structure: Information about the block structure
    """
    
    # --- 1. Generate Random System Matrices ---
    # State transition matrix G_k (n x n) - sparse
    G_k_mat = sprandn(T, n_state, n_state, sparsity_G)
    G_k_mat = G_k_mat + T(n_state)*I(n_state)  # Ensure numerical stability
    
    # Observation matrix H_k (m_obs x n_state) - sparse
    H_k_mat = sprandn(T, m_obs, n_state, sparsity_H)
    
    # Process noise covariance Q_k (n_state x n_state) - diagonal
    Q_k_mat = LinearAlgebra.diagm(0 => rand(T, n_state).^2 .+ T(0.1))
    
    # Observation noise covariance R_k (m_obs x m_obs) - diagonal
    R_k_mat = LinearAlgebra.diagm(0 => rand(T, m_obs).^2 .+ T(0.1))
    
    # --- 2. Construct Block Matrices ---
    total_state_dim = (N_timesteps + 1) * n_state
    total_obs_dim = N_timesteps * m_obs
    G_output_dim = N_timesteps * n_state
    
    # Inverse covariance matrices
    R_k_inv = inv(R_k_mat)
    Q_k_inv = inv(Q_k_mat)
    
    # Block diagonal R_inv_stacked
    R_inv_stacked = zeros(T, total_obs_dim, total_obs_dim)
    for k in 1:N_timesteps
        row_start = (k-1)*m_obs + 1
        row_end = k*m_obs
        R_inv_stacked[row_start:row_end, row_start:row_end] = R_k_inv
    end
    
    # Block diagonal Q_inv_stacked
    Q_inv_stacked = zeros(T, G_output_dim, G_output_dim)
    for k in 1:N_timesteps
        row_start = (k-1)*n_state + 1
        row_end = k*n_state
        Q_inv_stacked[row_start:row_end, row_start:row_end] = Q_k_inv
    end
    
    # H_stacked (block diagonal observation matrix)
    H_stacked = zeros(T, total_obs_dim, total_state_dim)
    for k in 1:N_timesteps
        row_start = (k-1)*m_obs + 1
        row_end = k*m_obs
        col_start = k*n_state + 1
        col_end = (k+1)*n_state
        H_stacked[row_start:row_end, col_start:col_end] = H_k_mat
    end
    
    # G_stacked (block bidiagonal state transition matrix)
    G_stacked = zeros(T, G_output_dim, total_state_dim)
    for k in 1:N_timesteps
        row_start = (k-1)*n_state + 1
        row_end = k*n_state
        
        # Diagonal I block (for x_k)
        col_start_I = k*n_state + 1
        col_end_I = (k+1)*n_state
        G_stacked[row_start:row_end, col_start_I:col_end_I] = I(n_state)
        
        # Sub-diagonal -G_k block (for x_{k-1})
        col_start_G = (k-1)*n_state + 1
        col_end_G = k*n_state
        G_stacked[row_start:row_end, col_start_G:col_end_G] = -G_k_mat
    end
    
    # --- 3. Construct System Matrix and RHS ---
    # System matrix A = H^T * R^{-1} * H + G^T * Q^{-1} * G
    LHS_H_term = H_stacked' * R_inv_stacked * H_stacked
    LHS_G_term = G_stacked' * Q_inv_stacked * G_stacked
    A = LHS_H_term + LHS_G_term
    
    # Generate random observations and prior
    z_stacked = randn(T, total_obs_dim)
    x0_prior = randn(T, n_state)
    zeta_adjusted = zeros(T, G_output_dim)
    zeta_adjusted[1:n_state] = x0_prior
    
    # RHS vector b = H^T * R^{-1} * z + G^T * Q^{-1} * zeta
    RHS_H_term = H_stacked' * R_inv_stacked * z_stacked
    RHS_G_term = G_stacked' * Q_inv_stacked * zeta_adjusted
    b = RHS_H_term + RHS_G_term
    
    block_structure = (
        n_state = n_state,
        m_obs = m_obs,
        N_timesteps = N_timesteps,
        total_state_dim = total_state_dim,
        sparsity = count(!iszero, A) / length(A)
    )
    
    return A, b, block_structure
end

function convert_to_block_tridiagonal_structure(A, b, n_state, N_timesteps)
    """
    Convert the Kalman Filter system matrix to block tridiagonal structure
    compatible with BlockStructuredSolvers.
    """
    N = N_timesteps + 1  # Number of blocks
    n = n_state
    
    # Extract diagonal blocks (A_i)
    A_list = Vector{Matrix{T}}(undef, N)
    for i in 1:N
        row_start = (i-1)*n + 1
        row_end = i*n
        A_list[i] = A[row_start:row_end, row_start:row_end]
    end
    
    # Extract off-diagonal blocks (B_i)
    B_list = Vector{Matrix{T}}(undef, N-1)
    for i in 1:N-1
        row_start = (i-1)*n + 1
        row_end = i*n
        col_start = i*n + 1
        col_end = (i+1)*n
        B_list[i] = A[row_start:row_end, col_start:col_end]
    end
    
    # Convert RHS to block structure
    d_list = Vector{Matrix{T}}(undef, N)
    for i in 1:N
        row_start = (i-1)*n + 1
        row_end = i*n
        d_list[i] = reshape(b[row_start:row_end], n, 1)
    end
    
    return A_list, B_list, d_list
end

function benchmark_cudss_kf(BigMatrix, d, N, n, backend_type)
    # Only run CUDSS if CUDA is available
    if !(CUDA.functional() && backend_type == CuArray)
        return nothing
    end
    
    try
        # Initialization time
        init_time = gpu_elapsed(backend_type) do
            BigMatrix_cudss = CUSPARSE.CuSparseMatrixCSR(BigMatrix)
            BigMatrix_cudss = CUDSS.CudssMatrix(BigMatrix_cudss, "SPD", 'U')
            config = CUDSS.CudssConfig()
            data = CUDSS.CudssData()
            solver = CUDSS.CudssSolver(BigMatrix_cudss, config, data)
            T_elem = eltype(BigMatrix)
            d_cudss = CuVector{T_elem}(vec(d))
            x_cudss = CUDA.zeros(T_elem, size(BigMatrix, 1))
            b_cudss = CUDA.zeros(T_elem, size(BigMatrix, 1))
            CUDSS.cudss("analysis", solver, x_cudss, b_cudss)
        end
        
        # Factorization
        factor_time = gpu_elapsed(backend_type) do
            CUDSS.cudss("factorization", solver, x_cudss, b_cudss)
        end
        
        # Solve
        solve_time = gpu_elapsed(backend_type) do
            CUDSS.cudss("solve", solver, x_cudss, d_cudss)
        end
        
        solution = Array(x_cudss)
        
        # Cleanup
        x_cudss = nothing
        d_cudss = nothing
        b_cudss = nothing
        BigMatrix_cudss = nothing
        solver = nothing
        config = nothing
        data = nothing
        gpu_reclaim(backend_type)
        
        return (init_time, factor_time, solve_time), solution
        
    catch e
        println("Warning: CUDSS failed: ", e)
        return nothing
    end
end

function fill_gpu_tensors_kf!(A_tensor_gpu, B_tensor_gpu, d_tensor_gpu, A_list_gpu, B_list_gpu, d_list_gpu, N, backend_type)
    gpu_allowscalar(backend_type) do
        for i in 1:N
            A_tensor_gpu[:, :, i] = A_list_gpu[i]
            d_tensor_gpu[:, :, i] = d_list_gpu[i]
        end
        for i in 1:N-1
            B_tensor_gpu[:, :, i] = B_list_gpu[i]
        end
    end
end

function benchmark_gpu_solver_kf(solver_type, A_tensor_gpu, B_tensor_gpu, d_tensor_gpu, N, n, backend_type)
    # Measure initialization time properly by timing everything
    start_time = time()
    
    if solver_type == :batched
        data = initialize_batched(N, n, T, backend_type)
    else  # sequential
        data = initialize_seq(N, n, T, backend_type)
    end
    
    copyto!(data.A_tensor, A_tensor_gpu)
    copyto!(data.B_tensor, B_tensor_gpu)
    copyto!(data.d_tensor, d_tensor_gpu)
    
    # Synchronize to ensure all operations are complete
    if backend_type == CuArray
        CUDA.synchronize()
    else
        AMDGPU.synchronize()
    end
    
    init_time = time() - start_time
    
    # Factorization
    factor_time = gpu_elapsed(backend_type) do
        gpu_sync(() -> factorize!(data), backend_type)
    end
    
    # Solve
    solve_time = gpu_elapsed(backend_type) do
        gpu_sync(() -> solve!(data), backend_type)
    end
    
    # Extract solution
    solution = vcat([vec(Array(data.d_list[i])) for i in 1:N]...)
    
    return (init_time, factor_time, solve_time), solution
end

function compute_residual_kf(solution, A_matrix, b_vector)
    """Compute residual ||Ax - b||_2 for the full system"""
    return norm(A_matrix * solution - b_vector, 2)
end

function run_kf_gpu(n_state, m_obs, N_timesteps)
    # Set random seed for reproducibility
    Random.seed!(seed)
    
    # Generate Kalman Filter problem
    println("Generating Kalman Filter problem...")
    A_full, b_full, block_info = generate_kalman_filter_problem(n_state, m_obs, N_timesteps)
    
    println("Problem info:")
    println("  State dimension: $(block_info.n_state)")
    println("  Observation dimension: $(block_info.m_obs)")
    println("  Time steps: $(block_info.N_timesteps)")
    println("  Total system size: $(block_info.total_state_dim)")
    println("  Sparsity: $(round(block_info.sparsity * 100, digits=2))%")
    
    # Convert to block tridiagonal structure
    A_list, B_list, d_list = convert_to_block_tridiagonal_structure(A_full, b_full, n_state, N_timesteps)
    N = length(A_list)
    n = n_state
    
    # Create GPU arrays
    if CUDA.functional()
        A_list_gpu, B_list_gpu, _, _, d_list_gpu = to_nvidia_gpu(A_list, B_list, [], [], d_list)
    else
        A_list_gpu, B_list_gpu, _, _, d_list_gpu = to_amd_gpu(A_list, B_list, [], [], d_list)
    end
    
    # Storage for results
    solutions = Dict{String, Vector{T}}()
    residuals = Dict{String, T}()
    timing_results = Dict{String, Tuple{Float64, Float64, Float64}}()
    
    # Create GPU tensors
    A_tensor_gpu = M{T}(undef, n, n, N)
    B_tensor_gpu = M{T}(undef, n, n, N-1)
    d_tensor_gpu = M{T}(undef, n, 1, N)
    
    # Fill GPU tensors
    fill_gpu_tensors_kf!(A_tensor_gpu, B_tensor_gpu, d_tensor_gpu, A_list_gpu, B_list_gpu, d_list_gpu, N, M)
    
    # Benchmark CUDSS (on full system)
    println("Running CUDSS benchmark...")
    cudss_result = benchmark_cudss_kf(sparse(A_full), b_full, N, n, M)
    if cudss_result !== nothing
        timing_results["CUDSS"], solutions["CUDSS"] = cudss_result
    end
    
    # Benchmark GPU Block Structured Solver
    println("Running GPU Block-Structured Solver benchmark...")
    timing_results["GPU_Batched"], solutions["GPU_Batched"] = 
        benchmark_gpu_solver_kf(:batched, A_tensor_gpu, B_tensor_gpu, d_tensor_gpu, N, n, M)
    
    # Compute residuals
    for (solver_name, solution) in solutions
        residuals[solver_name] = compute_residual_kf(solution, A_full, b_full)
    end
    
    # Cleanup
    gpu_reclaim(M)
    
    return timing_results, residuals, block_info
end

function write_kf_results_to_file(io, solver_name, timing_data, residual_data)
    init_times = [t[1] for t in timing_data]
    factor_times = [t[2] for t in timing_data]
    solve_times = [t[3] for t in timing_data]
    total_times = factor_times .+ solve_times
    
    println(io, @sprintf("  %s:", solver_name))
    println(io, @sprintf("    Init: %.6f ± %.6f ms", mean(init_times)*1000, std(init_times)*1000))
    println(io, @sprintf("    Factorize: %.6f ± %.6f ms", mean(factor_times)*1000, std(factor_times)*1000))
    println(io, @sprintf("    Solve: %.6f ± %.6f ms", mean(solve_times)*1000, std(solve_times)*1000))
    println(io, @sprintf("    Total: %.6f ± %.6f ms", mean(total_times)*1000, std(total_times)*1000))
    println(io, @sprintf("    Residual: %.2e ± %.2e", mean(residual_data), std(residual_data)))
    println(io)
end

function run_kf_benchmark_suite(problem_configs, iterations=10, output_file="kf_benchmark_results.txt")
    """
    Run Kalman Filter benchmarks for multiple problem configurations.
    
    problem_configs: Array of tuples (n_state, m_obs, N_timesteps)
    """
    
    all_results = []
    
    # Single warmup
    println("T = $T")
    println("M = $M")
    println("Running Kalman Filter warmup...")
    _, _, _ = run_kf_gpu(20, 2, 3)
    println("Warmup completed.\n")
    
    # Open output file in append mode
    open(output_file, "a") do io
        # Write header
        println(io, "\n" * "="^80)
        println(io, "NEW KALMAN FILTER BENCHMARK RUN")
        println(io, "GPU Kalman Filter Block Structured Solver Benchmark Results")
        println(io, "Generated: $(now())")
        println(io, "Data Type: $T")
        println(io, "GPU Backend: $(M == CuArray ? "CUDA" : "ROCm")")
        println(io, "Iterations per problem size: $iterations")
        println(io, "="^80)
        println(io)
        
        for (n_state, m_obs, N_timesteps) in problem_configs
            total_size = (N_timesteps + 1) * n_state
            println("Running KF benchmark for n_state=$n_state, m_obs=$m_obs, N_timesteps=$N_timesteps...")
            println(io, "Kalman Filter Problem: n_state=$n_state, m_obs=$m_obs, N_timesteps=$N_timesteps")
            println(io, "Total system size: $total_size")
            println(io, "-"^50)
            
            # Storage for this problem size
            all_timing_results = Dict{String, Vector{Tuple{Float64, Float64, Float64}}}()
            all_residuals = Dict{String, Vector{T}}()
            problem_info = nothing
            
            println("  Running $iterations iterations...")
            for i = 1:iterations
                timing_results, residuals, block_info = run_kf_gpu(n_state, m_obs, N_timesteps)
                problem_info = block_info
                
                # Collect results
                for (solver_name, timing) in timing_results
                    if !haskey(all_timing_results, solver_name)
                        all_timing_results[solver_name] = Tuple{Float64, Float64, Float64}[]
                    end
                    push!(all_timing_results[solver_name], timing)
                end
                
                for (solver_name, residual) in residuals
                    if !haskey(all_residuals, solver_name)
                        all_residuals[solver_name] = T[]
                    end
                    push!(all_residuals[solver_name], residual)
                end
            end
            
            # Write problem info
            if problem_info !== nothing
                println(io, "Problem characteristics:")
                println(io, "  Matrix sparsity: $(round(problem_info.sparsity * 100, digits=2))%")
                println(io)
            end
            
            # Write results for this problem size
            println(io, "Results:")
            for solver_name in sort(collect(keys(all_timing_results)))
                write_kf_results_to_file(io, solver_name, all_timing_results[solver_name], all_residuals[solver_name])
            end
            
            # Store for summary
            push!(all_results, (n_state, m_obs, N_timesteps, total_size, all_timing_results, all_residuals))
            println(io, "="^50 * "\n")
        end
        
        # Write summary table
        println(io, "KALMAN FILTER SUMMARY TABLE")
        println(io, "="^140)
        println(io, @sprintf("%-8s %-8s %-8s %-12s %-15s %-12s %-12s %-12s %-12s", 
               "n_state", "m_obs", "N_steps", "Total Size", "Solver", "Init (ms)", "Factor (ms)", "Solve (ms)", "Total (ms)"))
        println(io, "-"^140)
        
        for (n_state, m_obs, N_timesteps, total_size, timing_results, residuals) in all_results
            for solver_name in sort(collect(keys(timing_results)))
                timing_data = timing_results[solver_name]
                init_times = [t[1] for t in timing_data]
                factor_times = [t[2] for t in timing_data]
                solve_times = [t[3] for t in timing_data]
                total_times = factor_times .+ solve_times
                println(io, @sprintf("%-8d %-8d %-8d %-12d %-15s %-12.6f %-12.6f %-12.6f %-12.6f", 
                       n_state, m_obs, N_timesteps, total_size, solver_name, 
                       mean(init_times)*1000, mean(factor_times)*1000, 
                       mean(solve_times)*1000, mean(total_times)*1000))
            end
        end
    end
    
    println("Kalman Filter results saved to: $output_file")
end

# Example usage - Kalman Filter problem configurations
# Each tuple is (n_state, m_obs, N_timesteps)
kf_problem_configs = [
    # (50, 2, 10),    # Small problem
    # (100, 4, 8),    # Medium problem  
    # (200, 3, 6),    # Large problem
    # (256, 1024, 10),    # Very large problem
    (32, 128, 100),    # Very large problem
]

# Run the Kalman Filter benchmark suite
run_kf_benchmark_suite(kf_problem_configs, 10, "kf_benchmark_results.txt")
