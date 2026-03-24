using Test
using LinearAlgebra, SparseArrays

using RuizEquilibration

@testset "Dense nonsymmetric" begin
    A = [
        1e6 1e-3;
        1e-2 1e4
    ]
    eq = Equilibration(A)
    equilibrate!(eq)

    @test isscaled(eq)
    @test iterations(eq) > 0

    DAD = scaled_matrix(eq)
    D1 = left_scaling(eq)
    D2 = right_scaling(eq)

    # scaling identity: D1 * A * D2 == DAD
    @test D1 * A * D2 ≈ DAD

    # row and column inf-norms of DAD should be close to 1
    for i in axes(DAD, 1)
        @test maximum(abs, DAD[i, :]) ≈ 1 atol = 1e-6
    end
    for j in axes(DAD, 2)
        @test maximum(abs, DAD[:, j]) ≈ 1 atol = 1e-6
    end

    # functional interface
    D1f, DADf, D2f = equilibrate(A)
    @test D1f * A * D2f ≈ DADf
end

@testset "Dense symmetric" begin
    B = Symmetric([
        1e8 1.0;
        1.0 1e-4
    ])
    eq = SymmetricEquilibration(B)
    equilibrate!(eq)

    @test isscaled(eq)
    @test left_scaling(eq) === right_scaling(eq)

    DAD = scaled_matrix(eq)
    D = left_scaling(eq)
    @test D * parent(B) * D ≈ DAD

    # symmetry preserved
    @test DAD ≈ DAD'
end

@testset "Sparse nonsymmetric" begin
    A = sparse([
        1e4 0.0 1e-2;
        0.0 1e-3 0.0;
        1e2 0.0 1e6
    ])
    eq = Equilibration(A)
    equilibrate!(eq)

    @test isscaled(eq)

    DAD = scaled_matrix(eq)
    D1 = left_scaling(eq)
    D2 = right_scaling(eq)
    @test D1 * A * D2 ≈ DAD
    @test DAD isa SparseMatrixCSC
end

@testset "Sparse symmetric" begin
    S = sparse([
        1e6 0.0 1.0;
        0.0 1e-2 0.0;
        1.0 0.0 1e4
    ])
    eq = SymmetricEquilibration(Symmetric(S))
    equilibrate!(eq)

    @test isscaled(eq)
    @test left_scaling(eq) === right_scaling(eq)

    DAD = scaled_matrix(eq)
    D = left_scaling(eq)
    @test D * S * D ≈ DAD
end

@testset "Rectangular" begin
    A = [1e5 1e-1 1e3; 1e-4 1e2 1e-6]
    D1, DAD, D2 = equilibrate(A)
    @test D1 * A * D2 ≈ DAD
    @test size(DAD) == (2, 3)
end