@testset "Sequential cholesky factor" begin
    n = 128 # size of each block
    N = 550 # number of diagonal blocks

    A_list, B_list, x_list, x, d_list = generate_data(N, n)
    BigMatrix, d = construct_block_tridiagonal(A_list, B_list, d_list)
    N_blocks, block_size = detect_spaces_and_divide_csc(BigMatrix)

    for run in 2:10
        println("Run $run/10")
        
        #######################################
        A_list, B_list, x_list, x, d_list = generate_data(N, n)
        BigMatrix, d = construct_block_tridiagonal(A_list, B_list, d_list)
        
        # Time the detection of block size
        @time N_blocks, block_size = detect_spaces_and_divide_csc(BigMatrix)
        
        @test N_blocks == N
        @test block_size == n

    end
end