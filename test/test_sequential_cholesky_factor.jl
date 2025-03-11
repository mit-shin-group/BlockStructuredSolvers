@testset "Sequential cholesky factor" begin
    n = 100 # size of each block
    N = 55 # number of diagonal blocks

    # Run 3 times
    for run in 1:3
        println("Run $run/3")
        
        #######################################
        A_list = Vector{AbstractMatrix{Float64}}(undef, N)
        for i in 1:N
            temp = randn(n, n)
            A_list[i] = temp * temp' + n * I
        end

        B_list = Vector{AbstractMatrix{Float64}}(undef, N-1)
        for i in 1:N-1
            temp = randn(n, n)
            B_list[i] = temp
        end

        x_list = Vector{AbstractVector{Float64}}(undef, N)
        x = Vector{AbstractVector{Float64}}(undef, N)
        for i in 1:N
            x_list[i] = rand(n)
            x[i] = zeros(n)
        end

        d_list = Vector{AbstractVector{Float64}}(undef, N)
        d_list[1] = A_list[1] * x_list[1] + B_list[1] * x_list[2]
        @views for i = 2:N-1
            d_list[i] = B_list[i-1]' * x_list[i-1] + A_list[i] * x_list[i] + B_list[i] * x_list[i+1]
        end
        d_list[N] = B_list[N-1]' * x_list[N-1] + A_list[N] * x_list[N]

        #################################################

        ϵ = sqrt(eps(eltype(A_list[1])));

        data = initialize_sequential_cholesky_factor(N, n, A_list, B_list);

        GC.gc()
        println("  Factorization:")
        @time factorize_sequential_cholesky_factor!(data)
        
        println("  Solve:")
        @time solve_sequential_cholesky_factor!(data, d_list, x)

        @test norm(x - x_list) ≤ ϵ * norm(d_list)
    end
end