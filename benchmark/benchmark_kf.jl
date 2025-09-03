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

using JLD2   # lightweight, perfect for saving arrays & Dicts

"""
save_kf_plot_payload(path; t, X_true, X_hat, n_state, m_obs, N_timesteps, ρ, σq, σr,
                     solver_times=nothing, residuals=nothing)

Stores all info needed to plot x̂ vs x later without re-running.
"""
function save_kf_plot_payload(path::AbstractString; t, X_true, X_hat,
                              n_state::Int, m_obs::Int, N_timesteps::Int,
                              ρ::Float64, σq::Float64, σr::Float64,
                              solver_times=nothing, residuals=nothing)
    @save path t X_true X_hat n_state m_obs N_timesteps ρ σq σr solver_times residuals
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

# ---------- simulate time-invariant LGSSM with y_k = H x_k ----------
function simulate_lgssm(n::Int, m::Int, N::Int; ρ=0.98, σq=0.02, σr=0.10, rng=Random.default_rng())
    F = ρ * begin
        Q, _ = qr!(randn(rng, n, n)); M = Matrix(Q); det(M) < 0 && (M[:,1] .*= -1); M
    end
    H = randn(rng, m, n) ./ sqrt(n)

    U = begin
        Q, _ = qr!(randn(rng, n, n)); M = Matrix(Q); det(M) < 0 && (M[:,1] .*= -1); M
    end
    λ = range(0.5, 1.5; length=n)
    Qmat = (σq^2) * (U * Diagonal(collect(λ)) * U')
    Robj = (σr^2) * I(m)
    x0   = randn(rng, n)
    P0   = I(n) * 1.0

    X = zeros(eltype(Qmat), n, N+1)  # [x₀ x₁ … x_N]
    Y = zeros(eltype(Qmat), m, N)    # [y₁ … y_N]
    X[:,1] .= x0
    for k in 1:N
        X[:,k+1] .= F*X[:,k] + σq*randn(rng, n)          # x_{k} -> x_{k+1}
        Y[:,k]   .= H*X[:,k+1] + σr*randn(rng, m)        # *** y_k observes x_k ***
    end
    return (; F, H, Q=Qmat, R=Robj, P0, x0, X, Y)
end

# ---------- build A_list, B_list, d_list directly (no huge dense products) ----------
function build_kf_blocks(F::AbstractMatrix, H::AbstractMatrix,
                         Q::AbstractMatrix, R, P0, x0::AbstractVector, Y::AbstractMatrix)
    n, N = size(F,1), size(Y,2)

    # Precompute inverse actions once
    Qfac  = cholesky(Symmetric(Matrix(Q)))
    Qinv  = Qfac \ I
    QinvF = Qfac \ F

    # R^{-1} pieces
    HtRinvH, Rinv_y = if R isa UniformScaling
        rscale = 1/float(R.λ)
        (rscale * (H' * H), y -> rscale * y)
    else
        Rfac = cholesky(Symmetric(Matrix(R)))
        (H' * (Rfac \ H), y -> (Rfac \ y))
    end

    # P0^{-1}
    P0inv = P0 isa UniformScaling ? (1/float(P0.λ)) * I(n) :
             (cholesky(Symmetric(Matrix(P0))) \ I)

    FtQinvF = F' * QinvF
    B_up    = -(QinvF)'   # A_{k,k+1} = -F'Q^{-1} = (-(Q^{-1}F))'

    A_list = Vector{Matrix{eltype(F)}}(undef, N)
    B_list = Vector{Matrix{eltype(F)}}(undef, N-1)
    d_list = Vector{Matrix{eltype(F)}}(undef, N)

    for k in 1:N
        Akk = Matrix(HtRinvH)
        if k == 1
            Akk .+= P0inv .+ FtQinvF
        elseif k == N
            Akk .+= Qinv
        else
            Akk .+= Qinv .+ FtQinvF
        end
        A_list[k] = Symmetric(Akk) |> Matrix
        if k <= N-1
            B_list[k] = B_up
        end
        rhs_k = H' * Rinv_y(Y[:,k])
        if k == 1
            rhs_k .+= P0inv * x0
        end
        d_list[k] = reshape(rhs_k, n, 1)
    end
    return A_list, B_list, d_list
end

# ---------- convenience: simulate + blocks in one call ----------
function generate_kf_blocks_efficient(n_state::Int, m_obs::Int, N_timesteps::Int;
                                      ρ=0.98, σq=0.02, σr=0.10, rng=Random.default_rng())
    sim = simulate_lgssm(n_state, m_obs, N_timesteps; ρ=ρ, σq=σq, σr=σr, rng=rng)
    A_list, B_list, d_list = build_kf_blocks(sim.F, sim.H, sim.Q, sim.R, sim.P0, sim.x0, sim.Y)
    return (; A_list, B_list, d_list, sim...)
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

function run_kf_gpu(n_state, m_obs, N_timesteps; save_path::Union{Nothing,String}=nothing)
    Random.seed!(seed)

    # --- FAST PATH: directly build blocks; no giant dense A ---
    # (you currently hard-code ρ=0.98, σq=1.0, σr=5.0 here; we reuse them below)
    ρ_used, σq_used, σr_used = 0.98, 0.2, 1.0
    prob = generate_kf_blocks_efficient(n_state, m_obs, N_timesteps; ρ=ρ_used, σq=σq_used, σr=σr_used)
    A_list, B_list, d_list = prob.A_list, prob.B_list, prob.d_list
    N = length(A_list); n = n_state

    # Assemble sparse big matrix + rhs from blocks (your utils)
    BigMatrix, b_full = construct_block_tridiagonal(A_list, B_list, d_list)

    # For info printing (keep structure you had)
    block_info = (
        n_state = n_state,
        m_obs = m_obs,
        N_timesteps = N_timesteps,
        total_state_dim = N_timesteps * n_state,
        sparsity = nnz(BigMatrix) / length(BigMatrix)
    )

    # Move blocks to GPU (unchanged)
    if CUDA.functional()
        A_list_gpu, B_list_gpu, _, _, d_list_gpu = to_nvidia_gpu(A_list, B_list, [], [], d_list)
    else
        A_list_gpu, B_list_gpu, _, _, d_list_gpu = to_amd_gpu(A_list, B_list, [], [], d_list)
    end

    solutions = Dict{String, Vector{T}}()
    residuals = Dict{String, T}()
    timing_results = Dict{String, Tuple{Float64, Float64, Float64}}()

    # Tensors for our solver (unchanged)
    A_tensor_gpu = M{T}(undef, n, n, N)
    B_tensor_gpu = M{T}(undef, n, n, N-1)
    d_tensor_gpu = M{T}(undef, n, 1, N)
    fill_gpu_tensors_kf!(A_tensor_gpu, B_tensor_gpu, d_tensor_gpu,
                         A_list_gpu, B_list_gpu, d_list_gpu, N, M)

    # === cuDSS on sparse BigMatrix ===
    if CUDA.functional()
        println("Running CUDSS benchmark...")
        cudss_result = benchmark_cudss_kf(BigMatrix, b_full, N, n, CuArray)
        if cudss_result !== nothing
            timing_results["CUDSS"], solutions["CUDSS"] = cudss_result
        end
    end

    # === Your batched solver ===
    println("Running GPU Block-Structured Solver benchmark...")
    t0 = time()
    data = initialize_batched(N, n, T, M)
    copyto!(data.A_tensor, A_tensor_gpu)
    copyto!(data.B_tensor, B_tensor_gpu)
    copyto!(data.d_tensor, d_tensor_gpu)
    init_time = time() - t0

    factor_time = gpu_elapsed(() -> gpu_sync(() -> factorize!(data), M), M)
    solve_time  = gpu_elapsed(() -> gpu_sync(() -> solve!(data), M), M)

    # Collect solution as full vector
    solution_gpu = vcat([vec(Array(data.d_list[i])) for i in 1:N]...)
    timing_results["GPU_Batched"] = (init_time, factor_time, solve_time)
    solutions["GPU_Batched"] = solution_gpu

    # Residuals (use BigMatrix/b_full for both solvers)
    for (solver_name, x) in solutions
        residuals[solver_name] = norm(BigMatrix * x - b_full)
    end

    # -- optional save: everything needed for plotting x̂ vs x later --
    if save_path !== nothing
        t = collect(0:N)
        X_true = prob.X            # x1..xN
        X_hat  = reshape(solution_gpu, n, N)   # [x̂1 … x̂N]
        save_kf_plot_payload(save_path;
            t=t, X_true=X_true, X_hat=X_hat,
            n_state=n_state, m_obs=m_obs, N_timesteps=N_timesteps,
            ρ=ρ_used, σq=σq_used, σr=σr_used,
            solver_times=timing_results, residuals=residuals)
        @info "Saved KF payload for plotting" save_path
    end

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
    (256, 1024, 100),    # Very large problem
]

# Run the Kalman Filter benchmark suite
run_kf_benchmark_suite(kf_problem_configs, 10, "kf_benchmark_results.txt")

# Save the payload for plotting
# run_kf_gpu(256, 1024, 100; save_path="kf_payload_256x1024x100.jld2")