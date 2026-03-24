@testset "Equilibration - Constructor" begin
    @testset "float types" begin
        for T in (Float32, Float64)
            for (m, n) in ((5, 5), (4, 7), (7, 4))
                A = rand(T, m, n)
                eq = Equilibration(A)
                @test eq isa Equilibration{T, Matrix{T}}
                @test eq.DAD ≈ A
                @test eq.DAD !== A              # DAD is a copy
                @test size(eq.D1) == (m, m)
                @test size(eq.D2) == (n, n)
                @test length(eq.diag_row_inv) == m
                @test length(eq.diag_col_inv) == n
                @test length(eq.row_inf) == m
                @test length(eq.col_inf) == n
                @test eq.iter == 0
                @test eq.is_scaled == false
            end
        end
    end

    @testset "integer types convert to Float64" begin
        for T in (Int32, Int64, Bool)
            A = rand(T, 4, 6)
            eq = Equilibration(A)
            @test eq isa Equilibration{Float64, Matrix{Float64}}
            @test eq.DAD ≈ float(A)
        end
    end
end

@testset "Equilibration - equilibrate!" begin
    @testset "basic correctness (square)" begin
        A = Float64[2 0; 0 3]
        eq = Equilibration(A)
        A_orig = copy(A)
        equilibrate!(eq; tol=1e-8)

        @test isscaled(eq)
        @test iterations(eq) > 0
        @test all(isfinite, left_scaling(eq).diag)
        @test all(isfinite, right_scaling(eq).diag)
        @test all(>(0), left_scaling(eq).diag)
        @test all(>(0), right_scaling(eq).diag)

        # row and column infinity norms ≈ 1
        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(scaled_matrix(eq)))
        @test all(abs(1 - norm(col, Inf)) <= 1e-6 for col in eachcol(scaled_matrix(eq)))

        # fundamental scaling identity: D1 * A_orig * D2 = A_scaled
        @test left_scaling(eq) * A_orig * right_scaling(eq) ≈ scaled_matrix(eq)
    end

    @testset "rectangular matrix (more columns)" begin
        A = Float64[1 2 3 4; 5 6 7 8; 9 10 11 12]
        eq = Equilibration(A)
        A_orig = copy(A)
        equilibrate!(eq; tol=1e-8)

        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(scaled_matrix(eq)))
        @test all(abs(1 - norm(col, Inf)) <= 1e-6 for col in eachcol(scaled_matrix(eq)))
        @test left_scaling(eq) * A_orig * right_scaling(eq) ≈ scaled_matrix(eq)
    end

    @testset "rectangular matrix (more rows)" begin
        A = rand(8, 3)
        eq = Equilibration(A)
        A_orig = copy(A)
        equilibrate!(eq; tol=1e-8)

        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(scaled_matrix(eq)))
        @test all(abs(1 - norm(col, Inf)) <= 1e-6 for col in eachcol(scaled_matrix(eq)))
        @test left_scaling(eq) * A_orig * right_scaling(eq) ≈ scaled_matrix(eq)
    end

    @testset "ill-conditioned matrix" begin
        A = Float64[1e-8 0; 0 1e8]
        eq = Equilibration(A)
        A_orig = copy(A)
        equilibrate!(eq; tol=1e-8)

        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(scaled_matrix(eq)))
        @test all(abs(1 - norm(col, Inf)) <= 1e-6 for col in eachcol(scaled_matrix(eq)))
        @test left_scaling(eq) * A_orig * right_scaling(eq) ≈ scaled_matrix(eq)
    end

    @testset "idempotent — second call is a no-op" begin
        A = rand(5, 7)
        eq = Equilibration(A)
        equilibrate!(eq)
        DAD_1    = copy(scaled_matrix(eq))
        D1_diag1 = copy(left_scaling(eq).diag)
        D2_diag1 = copy(right_scaling(eq).diag)
        iters1   = iterations(eq)

        equilibrate!(eq)

        @test scaled_matrix(eq) ≈ DAD_1
        @test left_scaling(eq).diag == D1_diag1
        @test right_scaling(eq).diag == D2_diag1
        @test iterations(eq) == iters1
    end

    @testset "returns nothing" begin
        A = rand(4, 5)
        eq = Equilibration(A)
        @test equilibrate!(eq) === nothing
    end

    @testset "max_iter limits iterations" begin
        A = Float64[1 1000; 1 1]
        eq = Equilibration(A)
        equilibrate!(eq; max_iter=1)

        @test iterations(eq) == 1
        @test isscaled(eq)
    end

    @testset "float types preserved" begin
        for T in (Float32, Float64)
            A = rand(T, 6, 8)
            eq = Equilibration(A)
            equilibrate!(eq; tol=T(1e-5))

            @test eltype(scaled_matrix(eq)) == T
            @test eltype(left_scaling(eq).diag) == T
            @test eltype(right_scaling(eq).diag) == T
            @test all(abs(1 - norm(row, Inf)) <= 1e-4 for row in eachrow(scaled_matrix(eq)))
            @test all(abs(1 - norm(col, Inf)) <= 1e-4 for col in eachcol(scaled_matrix(eq)))
        end
    end

    @testset "already equilibrated converges in one iteration" begin
        A = Matrix{Float64}(I, 4, 4)
        eq = Equilibration(A)
        equilibrate!(eq; tol=1e-8)

        @test iterations(eq) == 1
        @test scaled_matrix(eq) ≈ A
    end
end

@testset "Equilibration - equilibrate (functional interface)" begin
    @testset "returns correct tuple" begin
        A = rand(5, 7)
        D1, DAD, D2 = equilibrate(A)

        @test D1 isa Diagonal
        @test D2 isa Diagonal
        @test size(DAD) == (5, 7)
        @test all(abs(1 - norm(row, Inf)) <= 1e-3 for row in eachrow(DAD))
        @test all(abs(1 - norm(col, Inf)) <= 1e-3 for col in eachcol(DAD))
    end

    @testset "scaling identity holds" begin
        A = rand(5, 7)
        D1, DAD, D2 = equilibrate(A)
        @test D1 * A * D2 ≈ DAD
    end

    @testset "original matrix is not modified" begin
        A = rand(5, 7)
        A_copy = copy(A)
        equilibrate(A)
        @test A == A_copy
    end
end

@testset "Equilibration - reset!" begin
    A = rand(4, 6)
    eq = Equilibration(A)
    equilibrate!(eq)

    @test isscaled(eq)

    RuizEquilibration.reset!(eq)

    @test !isscaled(eq)
    @test iterations(eq) == 0
    @test all(eq.D1.diag .== 1)
    @test all(eq.D2.diag .== 1)
    @test all(eq.diag_row_inv .== 1)
    @test all(eq.diag_col_inv .== 1)
    @test all(eq.row_inf .== 0)
    @test all(eq.col_inf .== 0)
end

@testset "Equilibration - update!" begin
    @testset "replaces matrix and resets state" begin
        A1 = rand(4, 6)
        A2 = rand(4, 6)
        eq = Equilibration(A1)
        equilibrate!(eq)

        @test isscaled(eq)

        update!(eq, A2)

        @test !isscaled(eq)
        @test iterations(eq) == 0
        @test scaled_matrix(eq) ≈ A2
    end

    @testset "can equilibrate after update" begin
        A1 = rand(4, 6)
        A2 = rand(4, 6)
        eq = Equilibration(A1)
        equilibrate!(eq)
        update!(eq, A2)
        equilibrate!(eq)

        @test isscaled(eq)
        @test all(abs(1 - norm(row, Inf)) <= 1e-3 for row in eachrow(scaled_matrix(eq)))
        @test all(abs(1 - norm(col, Inf)) <= 1e-3 for col in eachcol(scaled_matrix(eq)))
    end

    @testset "throws on wrong size" begin
        A = rand(4, 6)
        eq = Equilibration(A)
        @test_throws ArgumentError update!(eq, rand(3, 6))
        @test_throws ArgumentError update!(eq, rand(4, 5))
        @test_throws ArgumentError update!(eq, rand(5, 7))
    end
end

@testset "Equilibration - accessors" begin
    A = rand(5, 7)
    eq = Equilibration(A)

    @test scaled_matrix(eq) === eq.DAD
    @test left_scaling(eq) === eq.D1
    @test right_scaling(eq) === eq.D2
    @test iterations(eq) == 0
    @test !isscaled(eq)

    equilibrate!(eq)

    @test isscaled(eq)
    @test iterations(eq) > 0
end

@testset "Equilibration - special cases" begin
    @testset "zero matrix stays zero, no crash" begin
        A = zeros(3, 4)
        eq = Equilibration(A)
        equilibrate!(eq)

        @test all(iszero, scaled_matrix(eq))
        @test all(isfinite, left_scaling(eq).diag)
        @test all(isfinite, right_scaling(eq).diag)
        @test isscaled(eq)
    end

    @testset "zero row does not produce Inf or NaN" begin
        A = Float64[1 2 3; 0 0 0; 4 5 6]
        eq = Equilibration(A)
        equilibrate!(eq; tol=1e-8)

        @test all(isfinite, left_scaling(eq).diag)
        @test all(isfinite, right_scaling(eq).diag)
        @test all(isfinite, scaled_matrix(eq))
    end

    @testset "zero column does not produce Inf or NaN" begin
        A = Float64[1 0 3; 2 0 4; 5 0 6]
        eq = Equilibration(A)
        equilibrate!(eq; tol=1e-8)

        @test all(isfinite, left_scaling(eq).diag)
        @test all(isfinite, right_scaling(eq).diag)
        @test all(isfinite, scaled_matrix(eq))
    end

    @testset "1x1 matrix" begin
        A = reshape([5.0], 1, 1)
        eq = Equilibration(A)
        A_orig = copy(A)
        equilibrate!(eq; tol=1e-8)

        @test scaled_matrix(eq)[1, 1] ≈ 1.0
        @test left_scaling(eq) * A_orig * right_scaling(eq) ≈ scaled_matrix(eq)
    end

    @testset "single nonzero entry" begin
        A = zeros(5, 6)
        A[2, 4] = 1e-10
        eq = Equilibration(A)
        A_orig = copy(A)
        equilibrate!(eq; tol=1e-6)

        @test scaled_matrix(eq)[2, 4] ≈ 1.0 atol=1e-4
        @test all(isfinite, left_scaling(eq).diag)
        @test all(isfinite, right_scaling(eq).diag)
    end
end

@testset "Equilibration - random scaling identity" begin
    for _ in 1:20
        m, n = rand(3:10), rand(3:10)
        # matrix with varied magnitudes across several orders of magnitude
        A = rand(m, n) .* exp.(randn(m, n) .* 3)
        eq = Equilibration(A)
        A_orig = copy(A)
        equilibrate!(eq; tol=1e-8)

        D1  = left_scaling(eq)
        D2  = right_scaling(eq)
        DAD = scaled_matrix(eq)

        @test D1 * A_orig * D2 ≈ DAD
        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(DAD))
        @test all(abs(1 - norm(col, Inf)) <= 1e-6 for col in eachcol(DAD))
        @test all(>(0), D1.diag)
        @test all(>(0), D2.diag)
    end
end
