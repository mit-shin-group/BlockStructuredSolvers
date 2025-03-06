@testset "Sequential cholesky factor" begin
    n = 100 # size of each block
    N = 55 # number of diagonal blocks

    # Run 3 times
    for run in 1:3
        println("Run $run/3")
        
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

        data = initialize_sequential_cholesky_factor(N, n, A_list, B_list);
        x = zeros(data.N * n);

        GC.gc()
        println("  Factorization:")
        @time factorize_sequential_cholesky_factor!(data)
        
        println("  Solve:")
        @time solve_sequential_cholesky_factor!(data, d, x)

        @test norm(x - x_true) ≤ ϵ * norm(d)
    end
end