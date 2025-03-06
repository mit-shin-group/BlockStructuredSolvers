@testset "CUDA cholesky factor" begin
    n = 100 # size of each block
    m = 2 # number of blocks between separators
    P_start = 3 # number of separators
    level = 3

    # Run 3 times
    for run in 1:3
        println("Run $run/3")

        # Calculate N based on levels
        P = P_start
        N = P * (m + 1) - m
        for i = 2:level
            P = N
            N = P * (m + 1) - m
        end
        
        #######################################
        A_list = zeros(n, n, N);
        for i = 1:N
            temp = randn(Float64, n, n)
            A_list[:, :, i] = temp * temp' + n * I
        end

        B_list = zeros(n, n, N-1);
        for i = 1:N-1
            temp = randn(Float64, n, n)
            B_list[:, :, i] = temp
        end

        x_true = rand(N, n);
        d_list = zeros(N, n);
        d_list[1, :] = A_list[:, :, 1] * x_true[1, :] + B_list[:, :, 1] * x_true[2, :];

        @views for i = 2:N-1
            d_list[i, :] = B_list[:, :, i-1]' * x_true[i-1, :] + A_list[:, :, i] * x_true[i, :] + B_list[:, :, i] * x_true[i+1, :];
        end

        d_list[N, :] = B_list[:, :, N-1]' * x_true[N-1, :] + A_list[:, :, N] * x_true[N, :];

        d = zeros(N * n);

        @views for i = 1:N
            d[(i-1)*n+1:i*n] = d_list[i, :];
        end

        x_true = reshape(x_true', N*n);

        #################################################

        ϵ = sqrt(eps(eltype(A_list)));

        data = initialize_CUDA(P_start * (m + 1) - m, m, n, P_start, CuArray(A_list), CuArray(B_list), level);
        x = CuArray(zeros(data.N * n));
        x_true = CuArray(x_true);
        d = CuArray(d);

        GC.gc()
        println("  Factorization:")
        @time factorize_CUDA!(data)
        
        println("  Solve:")
        @time solve_CUDA!(data, d, x)

        @test norm(x - x_true) ≤ ϵ * norm(d)
    end
end