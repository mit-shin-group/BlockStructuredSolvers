@testset "Sequential cholesky factor" begin
    n = 100 # size of each block
    N = 55 # number of diagonal blocks

    A_list, B_list, x_list, x, d_list = generate_data(N, n)
    BigMatrix, d = construct_block_tridiagonal(A_list, B_list, d_list)
    N_blocks, block_size = detect_block_tridiagonal(BigMatrix)

    # Run 3 times
    for run in 2:10
        println("Run $run/3")
        
        #######################################
        A_list, B_list, x_list, x, d_list = generate_data(N, n)
        BigMatrix, d = construct_block_tridiagonal(A_list, B_list, d_list)
        
        # Time the detection of block size
        @time N_blocks, block_size = detect_block_tridiagonal(BigMatrix)
        
        @test N_blocks == N
        @test block_size == n

        #################################################

        # ϵ = sqrt(eps(eltype(A_list[1])));

        # data = initialize(N, n, A_list, B_list);

        # GC.gc()
        # println("  Factorization:")
        # @time factorize!(data)
        
        # println("  Solve:")
        # @time solve!(data, d_list, x)

        # @test mynorm(x - x_list) ≤ ϵ * mynorm(d_list)
    end
end