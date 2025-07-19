function _bss_norm(x)
    s = 0.0
    for i in eachindex(x)
        s += norm(x[i])^2
    end
    return sqrt(s)
end

function run_cpu_test(N, n, T)
    println("Testing CPU solver with $(T) precision")
    
    A_list = Vector{Matrix{T}}(undef, N);
    B_list = Vector{Matrix{T}}(undef, N-1);

    for i in 1:N
        temp = randn(n, n)
        A_list[i] = Matrix(temp * temp' + n * I) 
    end

    for i in 1:N-1
        temp = randn(n, n)
        B_list[i] = Matrix(temp)
    end

    d_list = Vector{Matrix{T}}(undef, N);

    x_list = Vector{Matrix{T}}(undef, N);
    for i in 1:N
        x_list[i] = Matrix{T}(rand(n, 1))
    end

    d_list[1] = A_list[1] * x_list[1] + B_list[1] * x_list[2];
    @views for i = 2:N-1
        d_list[i] = B_list[i-1]' * x_list[i-1] + A_list[i] * x_list[i] + B_list[i] * x_list[i+1]
    end
    d_list[N] = B_list[N-1]' * x_list[N-1] + A_list[N] * x_list[N];

    solver = initialize_cpu(N, n, T);

    for i in 1:N
        copyto!(solver.A_list[i], A_list[i])
    end

    for i in 1:N-1
        copyto!(solver.B_list[i], B_list[i])
    end

    for i in 1:N
        copyto!(solver.d_list[i], d_list[i])
    end
    
    # Calculate error tolerance based on precision
    ϵ = sqrt(eps(T))
    
    # Factorize
    println("  Factorization:")
    @time factorize!(solver)
    
    # Solve
    println("  Solve:")
    @time solve!(solver)
    
    # Compute error
    error_vec = [solver.d_list[i] - x_list[i] for i in 1:N]
    error = _bss_norm(error_vec)
    
    # Compute norm of the right-hand side for scaling
    rhs_norm = _bss_norm(x_list)
    
    # Test if error is within tolerance
    @test error ≤ ϵ * rhs_norm * 10
    
    # Also test that A*x_solved = b by computing the residual
    # Save the solution
    x_solved = solver.d_list

    # Compute the residual A*x_solved - b
    res_list = Vector{Matrix{T}}(undef, N)
    res_list[1] = A_list[1] * x_solved[1] + B_list[1] * x_solved[2] - d_list[1]
    @views for i = 2:N-1
        res_list[i] = B_list[i-1]' * x_solved[i-1] + 
                       A_list[i] * x_solved[i] + 
                       B_list[i] * x_solved[i+1] - 
                       d_list[i]
    end
    res_list[N] = B_list[N-1]' * x_solved[N-1] + A_list[N] * x_solved[N] - d_list[N]
    
    # Compute the residual norm
    res_norm = _bss_norm(res_list)
    
    # Test if residual is small
    @test res_norm ≤ ϵ * _bss_norm(d_list) * 10
    
    return error, rhs_norm, res_norm, _bss_norm(d_list)
end

@testset "Block structured CPU solvers" begin
    Random.seed!(42) # for reproducibility
    N = 50
    n = 32
    
    @testset "Float64 tests" begin
        error, rhs_norm, res_norm, d_norm = run_cpu_test(N, n, Float64)
        println("  Summary: ")
        println("    Relative solution error: $(error/rhs_norm)")
        println("    Relative residual: $(res_norm/d_norm)")
    end
    
    @testset "Float32 tests" begin
        error, rhs_norm, res_norm, d_norm = run_cpu_test(N, n, Float32)
        println("  Summary: ")
        println("    Relative solution error: $(error/rhs_norm)")
        println("    Relative residual: $(res_norm/d_norm)")
    end
end
