using LinearAlgebra, SparseArrays, BlockArrays
using BlockStructuredSolvers
using Plots
import Pkg
include("utils.jl")
using ProgressBars

gr() # Use GR backend
ENV["GKSwstype"] = "100" # Set GKS display type to file output only

# Fixed parameters
n = 10  # Size of each block
m = 2   # Number of blocks between separators
P = 3

# Test different problem sizes
level_values = [1, 2, 3, 4, 5]
factor_times = Float64[]
solve_times = Float64[]
N_values = Int[]

# Perform warmup run
println("Performing warmup run...")
level_warmup = level_values[1]  # Use smallest problem size for warmup
N_warmup = P * (m + 1) - m
BigMatrix_warmup, d_warmup, x_true_warmup, A_list_warmup, B_list_warmup = generate_tridiagonal_system(N_warmup, n)
data_warmup = initialize(N_warmup, m, n, P, A_list_warmup, B_list_warmup, level_warmup)
factorize!(data_warmup)
x_warmup = zeros(data_warmup.N * n)
solve!(data_warmup, d_warmup, x_warmup)

# Calculate N values
println("Running complexity analysis...")
for level in tqdm(level_values)

    N = P * (m + 1) - m;

    for i = 2:level

        P_temp = N;
        N = P_temp * (m + 1) - m;

    end

    # Generate test system
    BigMatrix, d, x_true, A_list, B_list = generate_tridiagonal_system(N, n)
    
    # Initialize solver
    data = initialize(P * (m + 1) - m, m, n, P, A_list, B_list, level)
    push!(N_values, data.N)
    
    # Measure factorization time
    factor_time = @elapsed factorize!(data)
    push!(factor_times, factor_time)
    
    # Measure solve time
    x = zeros(data.N * n)
    solve_time = @elapsed solve!(data, d, x)
    push!(solve_times, solve_time)
end

println("Generating plot...")

# Plot complexity graph
p1 = plot(N_values, factor_times, 
    label="Factorization",
    xlabel="Problem Size (N)", 
    ylabel="Time (s)",
    marker=:circle,
    yscale=:log10,
    xscale=:log10,
    title="Block Cholesky Performance",
    dpi=300)  # Increase DPI for better quality

plot!(p1, N_values, solve_times, 
    label="Solve",
    marker=:square)

# Save plot to file
savefig(p1, "block_cholesky_complexity.svg")
println("Plot saved to block_cholesky_complexity.svg")
