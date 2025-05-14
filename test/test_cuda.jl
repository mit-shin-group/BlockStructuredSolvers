using Test
using LinearAlgebra
using Random
using BlockStructuredSolvers
using CUDA

function run_test(N, n, T, solver_type)
    println("Testing $(solver_type) solver with $(T) precision")
    
    # Set up tensors based on solver type
    if solver_type == :batched
        MT = CuArray{T}
        
        # Initialize the solver
        data = initialize_batched(N, n, T)
        
        # Create test data
        A_tensor = MT(zeros(n, n, N))
        B_tensor = MT(zeros(n, n, N-1))
        
        # Generate positive definite A matrices and random B matrices
        CUDA.@allowscalar for i in 1:N
            temp = randn(T, n, n)
            A_tensor[:, :, i] .= MT(temp * temp' + n * I)
        end
        
        CUDA.@allowscalar for i in 1:N-1
            temp = randn(T, n, n)
            B_tensor[:, :, i] .= MT(temp)
        end
        
        # Copy data to solver
        copyto!(data.A_tensor, A_tensor)
        copyto!(data.B_tensor, B_tensor)
        
        # Create solution vector x and right-hand side d
        x_list = Vector{MT}(undef, N)
        for i in 1:N
            x_list[i] = MT(rand(T, n, 1))
        end
        
        # Compute right-hand side d = Ax
        d_tensor = MT(zeros(T, n, 1, N))
        d_tensor[:, :, 1] = A_tensor[:, :, 1] * x_list[1] + B_tensor[:, :, 1] * x_list[2]
        @views for i = 2:N-1
            d_tensor[:, :, i] = B_tensor[:, :, i-1]' * x_list[i-1] + A_tensor[:, :, i] * x_list[i] + B_tensor[:, :, i] * x_list[i+1]
        end
        d_tensor[:, :, N] = B_tensor[:, :, N-1]' * x_list[N-1] + A_tensor[:, :, N] * x_list[N]
        
        # Copy right-hand side to solver
        copyto!(data.d_tensor, d_tensor)
        
    elseif solver_type == :sequential
        MT = CuArray{T}
        
        # Initialize the solver
        data = initialize_seq(N, n, T)
        
        # Create test data
        A_tensor = MT(zeros(n, n, N))
        B_tensor = MT(zeros(n, n, N-1))
        
        # Generate positive definite A matrices and random B matrices
        CUDA.@allowscalar for i in 1:N
            temp = randn(T, n, n)
            A_tensor[:, :, i] .= MT(temp * temp' + n * I)
        end
        
        CUDA.@allowscalar for i in 1:N-1
            temp = randn(T, n, n)
            B_tensor[:, :, i] .= MT(temp)
        end
        
        # Copy data to solver
        copyto!(data.A_tensor, A_tensor)
        copyto!(data.B_tensor, B_tensor)
        
        # Create solution vector x and right-hand side d
        x_list = Vector{MT}(undef, N)
        for i in 1:N
            x_list[i] = MT(rand(T, n, 1))
        end
        
        # Compute right-hand side d = Ax
        d_tensor = MT(zeros(T, n, 1, N))
        d_tensor[:, :, 1] = A_tensor[:, :, 1] * x_list[1] + B_tensor[:, :, 1] * x_list[2]
        @views for i = 2:N-1
            d_tensor[:, :, i] = B_tensor[:, :, i-1]' * x_list[i-1] + A_tensor[:, :, i] * x_list[i] + B_tensor[:, :, i] * x_list[i+1]
        end
        d_tensor[:, :, N] = B_tensor[:, :, N-1]' * x_list[N-1] + A_tensor[:, :, N] * x_list[N]
        
        # Copy right-hand side to solver
        copyto!(data.d_tensor, d_tensor)
    end
    
    # Calculate error tolerance based on precision
    ϵ = sqrt(eps(T))
    
    # Factorize
    println("  Factorization:")
    CUDA.@sync factorize!(data)
    
    # Solve
    println("  Solve:")
    CUDA.@sync solve!(data)
    
    # Compute error
    error = 0.0
    CUDA.@allowscalar for i in 1:N
        error += norm(data.d_list[i] - x_list[i])^2
    end
    error = sqrt(error)
    
    # Compute norm of the right-hand side for scaling
    rhs_norm = 0.0
    for i in 1:N
        rhs_norm += norm(x_list[i])^2
    end
    rhs_norm = sqrt(rhs_norm)
    
    # Test if error is within tolerance
    @test error ≤ ϵ * rhs_norm * 10
    
    return error, rhs_norm
end

@testset "Block structured solvers" begin
    Random.seed!(42) # for reproducibility
    N = 50
    n = 32
    
    @testset "Float64 tests" begin
        @testset "Batched solver" begin
            error, rhs_norm = run_test(N, n, Float64, :batched)
            println("  Relative error: $(error/rhs_norm)")
        end
        
        @testset "Sequential solver" begin
            error, rhs_norm = run_test(N, n, Float64, :sequential)
            println("  Relative error: $(error/rhs_norm)")
        end
    end
    
    @testset "Float32 tests" begin
        @testset "Batched solver" begin
            error, rhs_norm = run_test(N, n, Float32, :batched)
            println("  Relative error: $(error/rhs_norm)")
        end
        
        @testset "Sequential solver" begin
            error, rhs_norm = run_test(N, n, Float32, :sequential)
            println("  Relative error: $(error/rhs_norm)")
        end
    end
end