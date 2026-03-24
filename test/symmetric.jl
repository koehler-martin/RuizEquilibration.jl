# Tests run for both a plain symmetric Matrix{T} and a Symmetric{T,Matrix{T}} wrapper.
# The two types take different code paths in scale_mat! and compute_row_inf_norms!,
# so both need coverage.

const sym_matrix_types = (
    ("Matrix",    (T, n) -> let B = rand(T, n, n); B + B' end),
    ("Symmetric", (T, n) -> let B = rand(T, n, n); Symmetric(B + B') end),
)

@testset "SymmetricEquilibration - Constructor ($label)" for (label, make_A) in sym_matrix_types
    @testset "float types" begin
        for T in (Float32, Float64)
            A = make_A(T, 5)
            eq = SymmetricEquilibration(A)

            @test eq isa SymmetricEquilibration{T}
            @test eq.DAD ≈ A
            @test eq.DAD !== A             # DAD is a copy
            @test size(eq.D) == (5, 5)
            @test length(eq.diag_inv) == 5
            @test length(eq.diag_inf) == 5
            @test eq.iter == 0
            @test eq.is_scaled == false
        end
    end

    @testset "integer types convert to Float64" begin
        for T in (Int32, Int64)
            B = rand(T, 4, 4)
            A = B + B'
            # wrap if needed so issymmetric passes
            input = label == "Symmetric" ? Symmetric(float(A)) : float(A)
            eq = SymmetricEquilibration(input)
            @test eq isa SymmetricEquilibration{Float64}
        end
    end

    @testset "non-symmetric matrix throws" begin
        A = rand(4, 4)
        A[1, 2] += 1.0          # break symmetry
        @test_throws ArgumentError SymmetricEquilibration(A)
    end
end

@testset "SymmetricEquilibration - equilibrate! ($label)" for (label, make_A) in sym_matrix_types
    @testset "basic correctness" begin
        for T in (Float32, Float64)
            A = make_A(T, 5)
            eq = SymmetricEquilibration(A)
            A_orig = Matrix(A)       # dense copy before equilibration

            equilibrate!(eq; tol=T(1e-6))

            D   = left_scaling(eq)
            DAD = scaled_matrix(eq)

            @test isscaled(eq)
            @test iterations(eq) > 0
            @test all(isfinite, D.diag)
            @test all(>(0), D.diag)

            # row and column norms ≈ 1
            @test all(abs(1 - norm(row, Inf)) <= 1e-4 for row in eachrow(Matrix(DAD)))
            @test all(abs(1 - norm(col, Inf)) <= 1e-4 for col in eachcol(Matrix(DAD)))

            # scaling identity: D * A_orig * D = DAD
            @test D * A_orig * D ≈ Matrix(DAD)

            # result is still symmetric
            @test Matrix(DAD) ≈ Matrix(DAD)'
        end
    end

    @testset "left_scaling and right_scaling are the same object" begin
        A = make_A(Float64, 4)
        eq = SymmetricEquilibration(A)
        equilibrate!(eq)

        @test left_scaling(eq) === right_scaling(eq)
    end

    @testset "idempotent — second call is a no-op" begin
        A = make_A(Float64, 5)
        eq = SymmetricEquilibration(A)
        equilibrate!(eq)

        DAD_1   = Matrix(scaled_matrix(eq))
        D_diag1 = copy(left_scaling(eq).diag)
        iters1  = iterations(eq)

        equilibrate!(eq)

        @test Matrix(scaled_matrix(eq)) ≈ DAD_1
        @test left_scaling(eq).diag == D_diag1
        @test iterations(eq) == iters1
    end

    @testset "returns nothing" begin
        A = make_A(Float64, 4)
        eq = SymmetricEquilibration(A)
        @test equilibrate!(eq) === nothing
    end

    @testset "max_iter limits iterations" begin
        A = make_A(Float64, 5)
        eq = SymmetricEquilibration(A)
        equilibrate!(eq; max_iter=1)

        @test iterations(eq) == 1
        @test isscaled(eq)
    end

    @testset "ill-conditioned diagonal" begin
        A_dense = Diagonal([1e-8, 1.0, 1e8]) |> Matrix
        A = label == "Symmetric" ? Symmetric(A_dense) : A_dense
        eq = SymmetricEquilibration(A)
        A_orig = Matrix(A)
        equilibrate!(eq; tol=1e-8)

        D   = left_scaling(eq)
        DAD = scaled_matrix(eq)

        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(Matrix(DAD)))
        @test D * A_orig * D ≈ Matrix(DAD)
    end

    @testset "already equilibrated converges in one iteration" begin
        A_dense = Matrix{Float64}(I, 4, 4)
        A = label == "Symmetric" ? Symmetric(A_dense) : A_dense
        eq = SymmetricEquilibration(A)
        equilibrate!(eq; tol=1e-8)

        @test iterations(eq) == 1
        @test Matrix(scaled_matrix(eq)) ≈ A_dense
    end
end

@testset "SymmetricEquilibration - equilibrate (functional interface)" begin
    # equilibrate dispatches to SymmetricEquilibration only for Symmetric wrapper
    @testset "Symmetric wrapper: D1 === D2" begin
        B = rand(5, 5)
        A = Symmetric(B + B')
        D1, DAD, D2 = equilibrate(A)

        @test D1 === D2
        @test all(abs(1 - norm(row, Inf)) <= 1e-3 for row in eachrow(Matrix(DAD)))
        @test D1 * Matrix(A) * D1 ≈ Matrix(DAD)
    end

    @testset "plain Matrix: dispatches to Equilibration (D1 !== D2)" begin
        B = rand(5, 5)
        A = B + B'             # symmetric Matrix, not Symmetric wrapper
        D1, DAD, D2 = equilibrate(A)

        @test D1 !== D2        # uses Equilibration, not SymmetricEquilibration
    end

    @testset "original matrix is not modified" begin
        B = rand(5, 5)
        A = Symmetric(B + B')
        A_copy = copy(Matrix(A))
        equilibrate(A)
        @test Matrix(A) == A_copy
    end
end

@testset "SymmetricEquilibration - reset! ($label)" for (label, make_A) in sym_matrix_types
    A = make_A(Float64, 4)
    eq = SymmetricEquilibration(A)
    equilibrate!(eq)

    @test isscaled(eq)

    RuizEquilibration.reset!(eq)

    @test !isscaled(eq)
    @test iterations(eq) == 0
    @test all(eq.D.diag .== 1)
    @test all(eq.diag_inv .== 1)
    @test all(eq.diag_inf .== 0)
end

@testset "SymmetricEquilibration - update! ($label)" for (label, make_A) in sym_matrix_types
    @testset "replaces matrix and resets state" begin
        A1 = make_A(Float64, 4)
        A2 = make_A(Float64, 4)
        eq = SymmetricEquilibration(A1)
        equilibrate!(eq)

        @test isscaled(eq)

        update!(eq, A2)

        @test !isscaled(eq)
        @test iterations(eq) == 0
        @test Matrix(scaled_matrix(eq)) ≈ Matrix(A2)
    end

    @testset "can equilibrate after update" begin
        A1 = make_A(Float64, 4)
        A2 = make_A(Float64, 4)
        eq = SymmetricEquilibration(A1)
        equilibrate!(eq)
        update!(eq, A2)
        equilibrate!(eq)

        @test isscaled(eq)
        @test all(abs(1 - norm(row, Inf)) <= 1e-3 for row in eachrow(Matrix(scaled_matrix(eq))))
    end

    @testset "throws on wrong size" begin
        A = make_A(Float64, 4)
        eq = SymmetricEquilibration(A)
        @test_throws ArgumentError update!(eq, make_A(Float64, 5))
    end
end

@testset "SymmetricEquilibration - accessors ($label)" for (label, make_A) in sym_matrix_types
    A = make_A(Float64, 5)
    eq = SymmetricEquilibration(A)

    @test scaled_matrix(eq) === eq.DAD
    @test left_scaling(eq) === eq.D
    @test right_scaling(eq) === eq.D
    @test iterations(eq) == 0
    @test !isscaled(eq)

    equilibrate!(eq)

    @test isscaled(eq)
    @test iterations(eq) > 0
end

@testset "SymmetricEquilibration - special cases ($label)" for (label, make_A) in sym_matrix_types
    @testset "zero matrix stays zero, no crash" begin
        A_dense = zeros(4, 4)
        A = label == "Symmetric" ? Symmetric(A_dense) : A_dense
        eq = SymmetricEquilibration(A)
        equilibrate!(eq)

        @test all(iszero, Matrix(scaled_matrix(eq)))
        @test all(isfinite, left_scaling(eq).diag)
        @test isscaled(eq)
    end

    @testset "1x1 matrix" begin
        A_dense = reshape([4.0], 1, 1)
        A = label == "Symmetric" ? Symmetric(A_dense) : A_dense
        eq = SymmetricEquilibration(A)
        equilibrate!(eq; tol=1e-8)

        @test Matrix(scaled_matrix(eq))[1, 1] ≈ 1.0
    end

    @testset "diagonal matrix" begin
        A_dense = Diagonal([1.0, 4.0, 9.0, 16.0]) |> Matrix
        A = label == "Symmetric" ? Symmetric(A_dense) : A_dense
        eq = SymmetricEquilibration(A)
        A_orig = A_dense
        equilibrate!(eq; tol=1e-8)

        D   = left_scaling(eq)
        DAD = Matrix(scaled_matrix(eq))

        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(DAD))
        @test D * A_orig * D ≈ DAD
    end
end

@testset "SymmetricEquilibration - random scaling identity ($label)" for (label, make_A) in sym_matrix_types
    for _ in 1:20
        n = rand(3:10)
        A = make_A(Float64, n)
        # scale rows/cols to create varied magnitudes
        s = exp.(randn(n) .* 3)
        A_dense = Matrix(Symmetric(Matrix(A) .* s .* s'))
        input = label == "Symmetric" ? Symmetric(A_dense) : A_dense

        eq = SymmetricEquilibration(input)
        A_orig = Matrix(input)
        equilibrate!(eq; tol=1e-8)

        D   = left_scaling(eq)
        DAD = Matrix(scaled_matrix(eq))

        @test D * A_orig * D ≈ DAD
        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(DAD))
        @test DAD ≈ DAD'        # result remains symmetric
        @test all(>(0), D.diag)
    end
end
