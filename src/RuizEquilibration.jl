"""
    RuizEquilibration

Matrix equilibration via the Ruiz algorithm.

Ruiz equilibration iteratively scales the rows and columns of a matrix so that
all infinity norms are close to 1. This is commonly used as a preprocessing
step before solving linear systems.

# Exports
- [`Equilibration`](@ref): scaling object for general (rectangular or nonsymmetric) matrices
- [`SymmetricEquilibration`](@ref): scaling object for symmetric matrices
- [`equilibrate!`](@ref): in-place equilibration
- [`equilibrate`](@ref): functional interface returning `(D1, DAD, D2)`
- [`scaled_matrix`](@ref), [`left_scaling`](@ref), [`right_scaling`](@ref): accessors
- [`iterations`](@ref), [`isscaled`](@ref), [`update!`](@ref): utilities

# References
- D. Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices",
  Technical Report RAL-TR-2001-034, Rutherford Appleton Laboratory, 2001.
"""
module RuizEquilibration
using LinearAlgebra, SparseArrays

export AbstractEquilibration
export iterations, isscaled, update!, equilibrate
export scaled_matrix, left_scaling, right_scaling
include("core.jl")

include("matrix_ops.jl")

export Equilibration, equilibrate!
include("nonsymmetric/type.jl")
include("nonsymmetric/util.jl")
include("nonsymmetric/equilibration.jl")

export SymmetricEquilibration
include("symmetric/type.jl")
include("symmetric/util.jl")
include("symmetric/equilibration.jl")

end
