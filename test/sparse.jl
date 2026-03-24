# Tests for sparse matrix support (SparseMatrixCSC).
# Exercises the CSC-specialised code paths in compute_inf_norms!,
# compute_col_inf_norms!, and scale_sym!.

# ──────────────────────────────────────────────────────────────────
# Nonsymmetric  (Equilibration with SparseMatrixCSC)
# ──────────────────────────────────────────────────────────────────

@testset "Sparse Equilibration - Constructor" begin
    for T in (Float32, Float64)
        for (m, n) in ((5, 5), (4, 7), (7, 4))
            A = sprand(T, m, n, 0.5)
            eq = Equilibration(A)
            @test eq isa Equilibration{T, SparseMatrixCSC{T, Int}}
            @test Matrix(eq.DAD) ≈ Matrix(A)
            @test eq.DAD !== A
            @test size(eq.D1) == (m, m)
            @test size(eq.D2) == (n, n)
            @test eq.iter == 0
            @test eq.is_scaled == false
        end
    end
end

@testset "Sparse Equilibration - equilibrate!" begin
    @testset "basic correctness (square)" begin
        A = sparse(Float64[2 0; 0 3])
        eq = Equilibration(A)
        A_orig = Matrix(A)
        equilibrate!(eq; tol=1e-8)

        @test isscaled(eq)
        @test iterations(eq) > 0

        DAD = Matrix(scaled_matrix(eq))
        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(DAD))
        @test all(abs(1 - norm(col, Inf)) <= 1e-6 for col in eachcol(DAD))
        @test left_scaling(eq) * A_orig * right_scaling(eq) ≈ DAD
    end

    @testset "rectangular (more columns)" begin
        A = sparse(Float64[1 2 3 4; 5 6 7 8; 9 10 11 12])
        eq = Equilibration(A)
        A_orig = Matrix(A)
        equilibrate!(eq; tol=1e-8)

        DAD = Matrix(scaled_matrix(eq))
        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(DAD))
        @test all(abs(1 - norm(col, Inf)) <= 1e-6 for col in eachcol(DAD))
        @test left_scaling(eq) * A_orig * right_scaling(eq) ≈ DAD
    end

    @testset "rectangular (more rows)" begin
        A = sparse(rand(8, 3))   # dense values in sparse format — no zero rows
        eq = Equilibration(A)
        A_orig = Matrix(A)
        equilibrate!(eq; tol=1e-8)

        DAD = Matrix(scaled_matrix(eq))
        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(DAD))
        @test all(abs(1 - norm(col, Inf)) <= 1e-6 for col in eachcol(DAD))
        @test left_scaling(eq) * A_orig * right_scaling(eq) ≈ DAD
    end

    @testset "ill-conditioned" begin
        A = sparse(Float64[1e-8 0; 0 1e8])
        eq = Equilibration(A)
        A_orig = Matrix(A)
        equilibrate!(eq; tol=1e-8)

        DAD = Matrix(scaled_matrix(eq))
        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(DAD))
        @test all(abs(1 - norm(col, Inf)) <= 1e-6 for col in eachcol(DAD))
        @test left_scaling(eq) * A_orig * right_scaling(eq) ≈ DAD
    end

    @testset "idempotent" begin
        A = sprand(5, 7, 0.5)
        eq = Equilibration(A)
        equilibrate!(eq)
        DAD_1    = Matrix(scaled_matrix(eq))
        D1_diag1 = copy(left_scaling(eq).diag)
        D2_diag1 = copy(right_scaling(eq).diag)
        iters1   = iterations(eq)

        equilibrate!(eq)

        @test Matrix(scaled_matrix(eq)) ≈ DAD_1
        @test left_scaling(eq).diag == D1_diag1
        @test right_scaling(eq).diag == D2_diag1
        @test iterations(eq) == iters1
    end

    @testset "max_iter limits iterations" begin
        A = sparse(Float64[1 1000; 1 1])
        eq = Equilibration(A)
        equilibrate!(eq; max_iter=1)

        @test iterations(eq) == 1
        @test isscaled(eq)
    end

    @testset "float types preserved" begin
        for T in (Float32, Float64)
            A = sparse(rand(T, 6, 8))
            eq = Equilibration(A)
            equilibrate!(eq; tol=T(1e-5))

            @test eltype(scaled_matrix(eq)) == T
            @test eltype(left_scaling(eq).diag) == T
            @test eltype(right_scaling(eq).diag) == T

            DAD = Matrix(scaled_matrix(eq))
            @test all(abs(1 - norm(row, Inf)) <= 1e-4 for row in eachrow(DAD))
            @test all(abs(1 - norm(col, Inf)) <= 1e-4 for col in eachcol(DAD))
        end
    end
end

@testset "Sparse Equilibration - equilibrate (functional)" begin
    @testset "returns correct tuple" begin
        A = sprand(5, 7, 0.5)
        D1, DAD, D2 = equilibrate(A)

        @test D1 isa Diagonal
        @test D2 isa Diagonal
        @test DAD isa SparseMatrixCSC
        @test size(DAD) == (5, 7)

        DAD_dense = Matrix(DAD)
        @test all(abs(1 - norm(row, Inf)) <= 1e-3 for row in eachrow(DAD_dense))
        @test all(abs(1 - norm(col, Inf)) <= 1e-3 for col in eachcol(DAD_dense))
    end

    @testset "scaling identity holds" begin
        A = sprand(5, 7, 0.5)
        D1, DAD, D2 = equilibrate(A)
        @test D1 * Matrix(A) * D2 ≈ Matrix(DAD)
    end

    @testset "original matrix is not modified" begin
        A = sprand(5, 7, 0.5)
        A_copy = copy(A)
        equilibrate(A)
        @test A == A_copy
    end
end

@testset "Sparse Equilibration - reset!" begin
    A = sprand(4, 6, 0.5)
    eq = Equilibration(A)
    equilibrate!(eq)

    @test isscaled(eq)

    RuizEquilibration.reset!(eq)

    @test !isscaled(eq)
    @test iterations(eq) == 0
    @test all(eq.D1.diag .== 1)
    @test all(eq.D2.diag .== 1)
end

@testset "Sparse Equilibration - update!" begin
    @testset "replaces matrix and resets state" begin
        A1 = sprand(4, 6, 0.5)
        A2 = sprand(4, 6, 0.5)
        eq = Equilibration(A1)
        equilibrate!(eq)

        @test isscaled(eq)

        update!(eq, A2)

        @test !isscaled(eq)
        @test iterations(eq) == 0
        @test Matrix(scaled_matrix(eq)) ≈ Matrix(A2)
    end

    @testset "can equilibrate after update" begin
        A1 = sparse(rand(4, 6))
        A2 = sparse(rand(4, 6))
        eq = Equilibration(A1)
        equilibrate!(eq)
        update!(eq, A2)
        equilibrate!(eq)

        @test isscaled(eq)
        DAD = Matrix(scaled_matrix(eq))
        @test all(abs(1 - norm(row, Inf)) <= 1e-3 for row in eachrow(DAD))
        @test all(abs(1 - norm(col, Inf)) <= 1e-3 for col in eachcol(DAD))
    end

    @testset "throws on wrong size" begin
        A = sprand(4, 6, 0.5)
        eq = Equilibration(A)
        @test_throws ArgumentError update!(eq, sprand(3, 6, 0.5))
        @test_throws ArgumentError update!(eq, sprand(4, 5, 0.5))
    end
end

@testset "Sparse Equilibration - special cases" begin
    @testset "zero matrix" begin
        A = spzeros(3, 4)
        eq = Equilibration(A)
        equilibrate!(eq)

        @test all(iszero, Matrix(scaled_matrix(eq)))
        @test all(isfinite, left_scaling(eq).diag)
        @test all(isfinite, right_scaling(eq).diag)
        @test isscaled(eq)
    end

    @testset "single nonzero entry" begin
        A = spzeros(5, 6)
        A[2, 4] = 1e-10
        eq = Equilibration(A)
        equilibrate!(eq; tol=1e-6)

        @test Matrix(scaled_matrix(eq))[2, 4] ≈ 1.0 atol=1e-4
        @test all(isfinite, left_scaling(eq).diag)
        @test all(isfinite, right_scaling(eq).diag)
    end
end

@testset "Sparse Equilibration - random scaling identity" begin
    for _ in 1:20
        m, n = rand(3:10), rand(3:10)
        # Use dense random values in sparse format to avoid zero rows/columns
        A = sparse(rand(m, n) .* exp.(randn(m, n) .* 3))
        eq = Equilibration(A)
        A_orig = Matrix(A)
        equilibrate!(eq; tol=1e-8)

        D1  = left_scaling(eq)
        D2  = right_scaling(eq)
        DAD = Matrix(scaled_matrix(eq))

        @test D1 * A_orig * D2 ≈ DAD
        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(DAD))
        @test all(abs(1 - norm(col, Inf)) <= 1e-6 for col in eachcol(DAD))
        @test all(>(0), D1.diag)
        @test all(>(0), D2.diag)
    end
end

# ──────────────────────────────────────────────────────────────────
# Symmetric  (SymmetricEquilibration with SparseMatrixCSC)
# ──────────────────────────────────────────────────────────────────

# Helper: build a structurally symmetric sparse matrix
function _make_sparse_sym(T, n; density=0.5)
    B = sprand(T, n, n, density)
    A = B + B'                      # symmetric, structurally symmetric
    return A
end

@testset "Sparse SymmetricEquilibration - Constructor" begin
    for T in (Float32, Float64)
        A = _make_sparse_sym(T, 5)
        eq = SymmetricEquilibration(A)

        @test eq isa SymmetricEquilibration{T, SparseMatrixCSC{T, Int}}
        @test Matrix(eq.DAD) ≈ Matrix(A)
        @test eq.DAD !== A
        @test size(eq.D) == (5, 5)
        @test eq.iter == 0
        @test eq.is_scaled == false
    end

    @testset "non-symmetric matrix throws" begin
        A = sprand(4, 4, 0.5)
        A[1, 2] += 1.0
        @test_throws ArgumentError SymmetricEquilibration(A)
    end
end

@testset "Sparse SymmetricEquilibration - equilibrate!" begin
    @testset "basic correctness" begin
        for T in (Float32, Float64)
            A = _make_sparse_sym(T, 6)
            eq = SymmetricEquilibration(A)
            A_orig = Matrix(A)
            equilibrate!(eq; tol=T(1e-6))

            D   = left_scaling(eq)
            DAD = Matrix(scaled_matrix(eq))

            @test isscaled(eq)
            @test iterations(eq) > 0
            @test all(isfinite, D.diag)
            @test all(>(0), D.diag)

            # row and column norms ≈ 1
            @test all(abs(1 - norm(row, Inf)) <= 1e-4 for row in eachrow(DAD))
            @test all(abs(1 - norm(col, Inf)) <= 1e-4 for col in eachcol(DAD))

            # scaling identity: D * A_orig * D = DAD
            @test D * A_orig * D ≈ DAD

            # result remains symmetric
            @test DAD ≈ DAD'
        end
    end

    @testset "left_scaling and right_scaling are the same object" begin
        A = _make_sparse_sym(Float64, 4)
        eq = SymmetricEquilibration(A)
        equilibrate!(eq)

        @test left_scaling(eq) === right_scaling(eq)
    end

    @testset "idempotent" begin
        A = _make_sparse_sym(Float64, 5)
        eq = SymmetricEquilibration(A)
        equilibrate!(eq)

        DAD_1  = Matrix(scaled_matrix(eq))
        D_diag = copy(left_scaling(eq).diag)
        iters1 = iterations(eq)

        equilibrate!(eq)

        @test Matrix(scaled_matrix(eq)) ≈ DAD_1
        @test left_scaling(eq).diag == D_diag
        @test iterations(eq) == iters1
    end

    @testset "returns nothing" begin
        A = _make_sparse_sym(Float64, 4)
        eq = SymmetricEquilibration(A)
        @test equilibrate!(eq) === nothing
    end

    @testset "max_iter limits iterations" begin
        A = _make_sparse_sym(Float64, 5)
        eq = SymmetricEquilibration(A)
        equilibrate!(eq; max_iter=1)

        @test iterations(eq) == 1
        @test isscaled(eq)
    end

    @testset "ill-conditioned diagonal" begin
        A = sparse(Diagonal([1e-8, 1.0, 1e8]) |> Matrix)
        A = A + A'  # ensure symmetric (already is, but consistent pattern)
        eq = SymmetricEquilibration(A)
        A_orig = Matrix(A)
        equilibrate!(eq; tol=1e-8)

        D   = left_scaling(eq)
        DAD = Matrix(scaled_matrix(eq))

        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(DAD))
        @test D * A_orig * D ≈ DAD
    end

    @testset "already equilibrated (identity)" begin
        A = sparse(Matrix{Float64}(I, 4, 4))
        eq = SymmetricEquilibration(A)
        equilibrate!(eq; tol=1e-8)

        @test iterations(eq) == 1
        @test Matrix(scaled_matrix(eq)) ≈ Matrix(I, 4, 4)
    end
end

@testset "Sparse SymmetricEquilibration - equilibrate (functional)" begin
    @testset "Symmetric wrapper dispatches correctly" begin
        A_sparse = _make_sparse_sym(Float64, 5)
        A_sym = Symmetric(A_sparse)
        D1, DAD, D2 = equilibrate(A_sym)

        @test D1 === D2

        DAD_dense = Matrix(DAD)
        @test all(abs(1 - norm(row, Inf)) <= 1e-3 for row in eachrow(DAD_dense))
        @test D1 * Matrix(A_sym) * D1 ≈ DAD_dense
    end

    @testset "plain symmetric sparse uses Equilibration (D1 !== D2)" begin
        A = _make_sparse_sym(Float64, 5)
        D1, DAD, D2 = equilibrate(A)

        @test D1 !== D2   # dispatches to nonsymmetric Equilibration
    end

    @testset "original matrix is not modified" begin
        A = _make_sparse_sym(Float64, 5)
        A_copy = copy(A)
        equilibrate(A)
        @test A == A_copy
    end
end

@testset "Sparse SymmetricEquilibration - reset!" begin
    A = _make_sparse_sym(Float64, 4)
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

@testset "Sparse SymmetricEquilibration - update!" begin
    @testset "replaces matrix and resets state" begin
        A1 = _make_sparse_sym(Float64, 4)
        A2 = _make_sparse_sym(Float64, 4)
        eq = SymmetricEquilibration(A1)
        equilibrate!(eq)

        @test isscaled(eq)

        update!(eq, A2)

        @test !isscaled(eq)
        @test iterations(eq) == 0
        @test Matrix(scaled_matrix(eq)) ≈ Matrix(A2)
    end

    @testset "can equilibrate after update" begin
        A1 = _make_sparse_sym(Float64, 4)
        A2 = _make_sparse_sym(Float64, 4)
        eq = SymmetricEquilibration(A1)
        equilibrate!(eq)
        update!(eq, A2)
        equilibrate!(eq)

        @test isscaled(eq)
        DAD = Matrix(scaled_matrix(eq))
        @test all(abs(1 - norm(row, Inf)) <= 1e-3 for row in eachrow(DAD))
    end

    @testset "throws on wrong size" begin
        A = _make_sparse_sym(Float64, 4)
        eq = SymmetricEquilibration(A)
        @test_throws ArgumentError update!(eq, _make_sparse_sym(Float64, 5))
    end
end

@testset "Sparse SymmetricEquilibration - special cases" begin
    @testset "zero matrix" begin
        A = spzeros(4, 4)
        eq = SymmetricEquilibration(A)
        equilibrate!(eq)

        @test all(iszero, Matrix(scaled_matrix(eq)))
        @test all(isfinite, left_scaling(eq).diag)
        @test isscaled(eq)
    end

    @testset "1x1 matrix" begin
        A = sparse([4.0][:, :])   # 1×1 sparse
        eq = SymmetricEquilibration(A)
        equilibrate!(eq; tol=1e-8)

        @test Matrix(scaled_matrix(eq))[1, 1] ≈ 1.0
    end

    @testset "diagonal matrix" begin
        A = sparse(Diagonal([1.0, 4.0, 9.0, 16.0]) |> Matrix)
        eq = SymmetricEquilibration(A)
        A_orig = Matrix(A)
        equilibrate!(eq; tol=1e-8)

        D   = left_scaling(eq)
        DAD = Matrix(scaled_matrix(eq))

        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(DAD))
        @test D * A_orig * D ≈ DAD
    end
end

@testset "Sparse SymmetricEquilibration - random scaling identity" begin
    for _ in 1:20
        n = rand(3:10)
        A = _make_sparse_sym(Float64, n)
        s = exp.(randn(n) .* 3)
        A_scaled = sparse(Symmetric(Matrix(A) .* s .* s'))

        eq = SymmetricEquilibration(A_scaled)
        A_orig = Matrix(A_scaled)
        equilibrate!(eq; tol=1e-8)

        D   = left_scaling(eq)
        DAD = Matrix(scaled_matrix(eq))

        @test D * A_orig * D ≈ DAD
        @test all(abs(1 - norm(row, Inf)) <= 1e-6 for row in eachrow(DAD))
        @test DAD ≈ DAD'
        @test all(>(0), D.diag)
    end
end

# ──────────────────────────────────────────────────────────────────
# Cross-check: sparse and dense give the same result
# ──────────────────────────────────────────────────────────────────

@testset "Sparse vs Dense agreement" begin
    @testset "nonsymmetric" begin
        A_dense = rand(6, 8) .* exp.(randn(6, 8) .* 2)
        A_sparse = sparse(A_dense)

        D1d, DADd, D2d = equilibrate(A_dense;  tol=1e-10)
        D1s, DADs, D2s = equilibrate(A_sparse; tol=1e-10)

        @test D1d.diag ≈ D1s.diag
        @test D2d.diag ≈ D2s.diag
        @test DADd     ≈ Matrix(DADs)
    end

    @testset "symmetric" begin
        B = rand(5, 5)
        A_dense = Symmetric(B + B')
        A_sparse = Symmetric(sparse(Matrix(A_dense)))

        D1d, DADd, D2d = equilibrate(A_dense;  tol=1e-10)
        D1s, DADs, D2s = equilibrate(A_sparse; tol=1e-10)

        @test D1d.diag ≈ D1s.diag
        @test Matrix(DADd) ≈ Matrix(DADs)
    end
end
