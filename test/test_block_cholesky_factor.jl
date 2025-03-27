@testset "Block cholesky factor" begin
    N = 97
    n = 32 # size of each block
    # m = 2 # number of blocks between separators
    # P_start = 3 # number of separators
    # level = 3

    # # Calculate N based on levels
    # P = P_start
    # N = P * (m + 1) - m
    # for i = 2:level
    #     P = N
    #     N = P * (m + 1) - m
    # end

    # Run 3 times
    for run in 1:3
        println("Run $run/3")
        
        #######################################
        A_list = Vector{Matrix{Float64}}(undef, N)
        for i in 1:N
            temp = randn(n, n)
            A_list[i] = temp * temp' + n * I
        end

        B_list = Vector{Matrix{Float64}}(undef, N-1)
        for i in 1:N-1
            temp = randn(n, n)
            B_list[i] = temp
        end

        x_list = Vector{Matrix{Float64}}(undef, N)
        x = Vector{Matrix{Float64}}(undef, N)
        for i in 1:N
            x_list[i] = rand(n, 1)
            x[i] = zeros(n, 1)
        end

        d_list = Vector{Matrix{Float64}}(undef, N)
        d_list[1] = A_list[1] * x_list[1] + B_list[1] * x_list[2]
        @views for i = 2:N-1
            d_list[i] = B_list[i-1]' * x_list[i-1] + A_list[i] * x_list[i] + B_list[i] * x_list[i+1]
        end
        d_list[N] = B_list[N-1]' * x_list[N-1] + A_list[N] * x_list[N]

        #################################################

        ϵ = sqrt(eps(eltype(A_list[1])));

        data = initialize(N, n, A_list, B_list, parallel_factorization = true);

        GC.gc()
        println("  Factorization:")
        @time factorize!(data)
        
        println("  Solve:")
        @time solve!(data, d_list, x)

        @test _bss_norm(x - x_list) ≤ ϵ * _bss_norm(d_list)
    end
end

if true #@isdefined(cuda_enabled) && cuda_enabled
    @testset "Block cholesky factor (CUDA)" begin
        N = 97
        n = 32 # size of each block

        # Run 3 times
        for run in 1:3
            println("Run $run/3")
            
            #######################################
            A_list = Vector{CuMatrix{Float64, DeviceMemory}}(undef, N)
            for i in 1:N
                temp = randn(n, n)
                A_list[i] = CuMatrix(temp * temp' + n * I)
            end
            
            B_list = Vector{CuMatrix{Float64, DeviceMemory}}(undef, N-1)
            for i in 1:N-1
                temp = randn(n, n)
                B_list[i] = CuMatrix(temp)
            end
            
            x_list = Vector{CuMatrix{Float64, DeviceMemory}}(undef, N)
            x = Vector{CuMatrix{Float64, DeviceMemory}}(undef, N)
            for i in 1:N
                x_list[i] = CuMatrix(rand(n, 1))
                x[i] = CuMatrix(zeros(n, 1))
            end
            
            d_list = Vector{CuMatrix{Float64, DeviceMemory}}(undef, N)
            d_list[1] = A_list[1] * x_list[1] + B_list[1] * x_list[2]
            @views for i = 2:N-1
                d_list[i] = B_list[i-1]' * x_list[i-1] + A_list[i] * x_list[i] + B_list[i] * x_list[i+1]
            end
            d_list[N] = B_list[N-1]' * x_list[N-1] + A_list[N] * x_list[N]

            #################################################

            ϵ = sqrt(eps(eltype(A_list[1])));

            data = initialize(N, n, A_list, B_list);

            GC.gc()
            println("  Factorization:")
            @time factorize!(data)
            
            println("  Solve:")
            @time solve!(data, d_list, x)

            @test _bss_norm(x - x_list) ≤ ϵ * _bss_norm(d_list)
        end
    end
else
    @info "Skipping CUDA tests for Block cholesky factor on this platform"
end